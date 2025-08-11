// -*- coding: utf-8 -*-
// RuoyiJarNormalizeSmart.java  (FIXED: final 赋值 & BasicFileAttributes import)
//
// 用途：桥接 ruoyi 风格的 CalculateCaptcha.jar，先调用其 -n/-p 生成原图，
//      再按给定的尺寸分布高质量归一化到目标尺寸（等比缩放 + 居中留白），
//      输出统一 images/<outDir>/images/*.png 和 labels.txt，并打印统计。
// 编译：
//   javac -encoding UTF-8 RuoyiJarNormalizeSmart.java
// 运行：
//   java -D"file.encoding=UTF-8" -cp ".;CalculateCaptcha-1.1.jar" RuoyiJarNormalizeSmart smart ..\outputs\ruoyi_train 120000 --jar CalculateCaptcha-1.1.jar --sizes "160x60:40,180x64:20,200x64:15,256x64:10,120x40:10,96x32:5" --threads 8
//

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;   // ✅ 补充这个 import
import java.util.List;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class RuoyiJarNormalizeSmart {

    public static void main(String[] args) {
        if (args.length == 0 || "help".equalsIgnoreCase(args[0])) { printHelp(); return; }
        try{
            switch (args[0].toLowerCase(Locale.ROOT)){
                case "smart": doSmart(Arrays.copyOfRange(args,1,args.length)); break;
                default:
                    System.out.println("未知命令: " + args[0]);
                    printHelp();
            }
        }catch(Exception e){
            e.printStackTrace();
            System.exit(2);
        }
    }

    static void printHelp(){
        System.out.println("用法：");
        System.out.println("  smart <outDir> <total> [--jar CalculateCaptcha-1.1.jar] [--sizes \"160x60:40,180x64:20,200x64:15,256x64:10,120x40:10,96x32:5\"] [--threads 8]");
        System.out.println("说明：先调用 ruoyi 的 JAR 生成原始图片，再按尺寸分布归一化到目标尺寸。");
    }

    /* ========== 参数解析 & 小工具 ========== */
    static Map<String,String> parseOpts(String[] a, int from){
        Map<String,String> m = new LinkedHashMap<>();
        for (int i=from;i<a.length;i++){
            String k=a[i];
            if (k.startsWith("--")){
                String key=k.substring(2);
                String val=(i+1<a.length && !a[i+1].startsWith("--"))? a[++i] : "1";
                m.put(key,val);
            }
        }
        return m;
    }

    static String safeName(String s){
        if (s==null) return "out";
        String r = s.replaceAll("[\\\\/:*?\"<>|\\s]+","_").replaceAll("_+","_");
        if (r.startsWith("_")) r=r.substring(1);
        if (r.isEmpty()) r="out";
        return r;
    }

    static final String DEFAULT_SIZES = "160x60:40,180x64:20,200x64:15,256x64:10,120x40:10,96x32:5";

    static class SizeBucket {
        final int W,H; final double w;
        SizeBucket(int W,int H,double w){ this.W=W; this.H=H; this.w=w; }
        String key(){ return W+"x"+H; }
    }

    // 原始权重容器（非 final，可在归一化前修改）
    static class RawSize { int W,H; double w; RawSize(int W,int H,double w){ this.W=W; this.H=H; this.w=w; } }

    static List<SizeBucket> parseSizes(String s){
        if (s==null || s.trim().isEmpty()) s = DEFAULT_SIZES;

        // 先用 RawSize 收集
        List<RawSize> tmp = new ArrayList<>();
        for (String it : s.split(",")){
            it = it.trim(); if (it.isEmpty()) continue;
            String[] kv = it.split(":");
            String[] wh = kv[0].toLowerCase(Locale.ROOT).split("x");
            int W = Integer.parseInt(wh[0].trim());
            int H = Integer.parseInt(wh[1].trim());
            double w = (kv.length>1)? Double.parseDouble(kv[1].trim()) : 1.0;
            tmp.add(new RawSize(W,H,Math.max(0,w)));
        }

        double sum=0; for (RawSize r:tmp) sum += r.w;
        if (sum<=0) { // 全是 0，则平均分配
            for (RawSize r:tmp) r.w = 1.0;
            sum = tmp.size();
        }

        // 归一化到 SizeBucket（final 字段，不再修改）
        List<SizeBucket> out = new ArrayList<>();
        for (RawSize r:tmp){
            out.add(new SizeBucket(r.W, r.H, r.w/sum));
        }
        return out;
    }

    static List<int[]> planCounts(List<SizeBucket> buckets, int total){
        List<int[]> plan = new ArrayList<>();
        int assigned=0;
        for (int i=0;i<buckets.size();i++){
            SizeBucket b=buckets.get(i);
            int c = (i==buckets.size()-1)? (total-assigned) : (int)Math.round(total*b.w);
            if (c<0) c=0;
            assigned += c;
            plan.add(new int[]{b.W,b.H,c});
        }
        if (assigned!=total){
            int[] last = plan.get(plan.size()-1);
            last[2] += (total-assigned);
        }
        return plan;
    }

    static String uuid(){ return java.util.UUID.randomUUID().toString().replace("-", ""); }

    /* ========== 主流程 ========== */
    static void doSmart(String[] argv) throws Exception {
        if (argv.length < 2){
            System.out.println("参数不足。用法：smart <outDir> <total> [--jar CalculateCaptcha-1.1.jar] [--sizes \""+DEFAULT_SIZES+"\"] [--threads 8]");
            return;
        }
        String outName = safeName(argv[0]);
        int total = Integer.parseInt(argv[1]);
        Map<String,String> opt = parseOpts(argv, 2);
        String jarPath = opt.getOrDefault("jar", "CalculateCaptcha-1.1.jar");
        String sizesArg = opt.getOrDefault("sizes", DEFAULT_SIZES);
        int threads = Integer.parseInt(opt.getOrDefault("threads", String.valueOf(Math.max(1, Runtime.getRuntime().availableProcessors()))));

        List<SizeBucket> sizeBuckets = parseSizes(sizesArg);
        List<int[]> plan = planCounts(sizeBuckets, total);

        Path outDir  = Paths.get("images").resolve(outName);
        Path outImgs = outDir.resolve("images");
        Path outLbl  = outDir.resolve("labels.txt");
        Files.createDirectories(outImgs);

        System.out.printf(">> RuoYi smart -> %s  (total=%d)\n", outDir, total);
        System.out.println("   sizes = " + sizesArg);
        System.out.println("   threads = " + threads);

        // 1) 调 ruoyi JAR 生成原始图片到临时前缀
        String prefix = "cap_calc_" + System.currentTimeMillis();
        runRuoyiJar(jarPath, total, prefix);

        // 2) 找到输出目录（ruoyi JAR 会在 datasets/<prefix> 下）
        Path dsDir1 = Paths.get("datasets").resolve(prefix);
        Path dsDir2 = Paths.get(prefix);
        Path srcDir = Files.isDirectory(dsDir1) ? dsDir1 : (Files.isDirectory(dsDir2) ? dsDir2 : null);
        if (srcDir==null) throw new IOException("未找到 ruoyi 生成目录：datasets/"+prefix+" 或 ./" + prefix);

        // 3) 收集源文件
        List<Path> srcFiles = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(srcDir, "*.{jpg,jpeg,png,JPG,JPEG,PNG}")) {
            for (Path p : ds) srcFiles.add(p);
        } catch (DirectoryIteratorException e) {
            e.printStackTrace();
        }
        if (srcFiles.isEmpty()) throw new IOException("ruoyi 生成目录为空：" + srcDir);

        // 4) 预分配每个目标尺寸应生成的数量
        Queue<Path> queue = new ArrayDeque<>(srcFiles);
        List<int[]> quota = new ArrayList<>(plan); // [W,H,count]

        // 统计
        Map<String,Integer> statSize  = new ConcurrentHashMap<>();
        Map<String,Integer> statLen   = new ConcurrentHashMap<>();
        AtomicInteger idx = new AtomicInteger(0);

        ExecutorService pool = Executors.newFixedThreadPool(threads);
        final Object labelLock = new Object();

        try(BufferedWriter bw = Files.newBufferedWriter(outLbl, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)){

            for (int[] q : quota){
                final int W=q[0], H=q[1], need=q[2];
                for (int i=0;i<need;i++){
                    final Path src = queue.poll();
                    if (src==null){
                        // 若 ruoyi 实际产出少于 total，则循环利用
                        break;
                    }
                    pool.submit(() -> {
                        try{
                            String label = extractLabelFromFilename(src.getFileName().toString());
                            BufferedImage img = ImageIO.read(src.toFile());
                            BufferedImage out = fitIntoCanvas(img, W, H, Color.WHITE);

                            int n = idx.incrementAndGet();
                            String fn = String.format("%08d.png", n);
                            ImageIO.write(out, "png", outImgs.resolve(fn).toFile());

                            synchronized (labelLock){
                                bw.write(fn); bw.write('\t'); bw.write(label); bw.newLine();
                            }

                            statInc(statSize, W+"x"+H);
                            statInc(statLen, String.valueOf(measureTextLen(label)));

                            if (n % 200 == 0 || n==total){
                                System.out.printf("\r.. %d/%d", n, total);
                            }
                        }catch(Exception e){
                            e.printStackTrace();
                        }
                    });
                }
            }

            pool.shutdown();
            pool.awaitTermination(3, TimeUnit.DAYS);
        }

        // 5) 清理 ruoyi 的临时目录
        try { deleteDirRecursive(srcDir); } catch(Exception ignore){}

        // 6) 打印统计
        System.out.println("\n✅ 完成：" + outDir.toAbsolutePath());
        System.out.println("== 尺寸分布（目标） ==");
        printStat(statSize, total);
        System.out.println("\n== 文本长度分布（观测） ==");
        printStat(statLen, total);
    }

    /* ========== ruoyi JAR 调用 ========== */
    static void runRuoyiJar(String jarPath, int total, String prefix) throws Exception {
        List<String> cmd = new ArrayList<>();
        cmd.add(detectJavaExe());
        cmd.add("-jar");
        cmd.add(jarPath);
        cmd.add("-n"); cmd.add(String.valueOf(total));
        cmd.add("-p"); cmd.add(prefix);

        System.out.println(".. 调用: " + String.join(" ", cmd));
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(true);
        Process ps = pb.start();

        try (BufferedReader br = new BufferedReader(new InputStreamReader(ps.getInputStream(), detectConsoleCharset()))) {
            String line;
            while ((line = br.readLine()) != null) {
                System.out.println(line);
            }
        }

        int code = ps.waitFor();
        if (code != 0) throw new RuntimeException("调用 ruoyi JAR 失败，退出码=" + code);
    }

    static String detectJavaExe(){
        String javaHome = System.getenv("JAVA_HOME");
        if (javaHome!=null && !javaHome.isEmpty()){
            Path p = Paths.get(javaHome,"bin","java.exe");
            if (Files.isRegularFile(p)) return p.toString();
        }
        return "java";
    }

    static Charset detectConsoleCharset(){
        try{
            return Charset.forName(System.getProperty("sun.jnu.encoding","UTF-8"));
        }catch(Exception e){ return StandardCharsets.UTF_8; }
    }

    /* ========== 标签解析 / 归一化缩放 ========== */
    static String extractLabelFromFilename(String name){
        // ruoyi 生成形如：  "7×3=？_hash.jpg"
        int p = name.indexOf('_');
        String base = (p>0) ? name.substring(0,p) : name;
        int dot = base.lastIndexOf('.');
        if (dot>0) base = base.substring(0,dot);
        return base;
    }

    static int measureTextLen(String s){
        return s!=null ? s.length() : 0;
    }

    static BufferedImage fitIntoCanvas(BufferedImage src, int W, int H, Color bg){
        int sw = src.getWidth(), sh = src.getHeight();
        double sx = (double) (W) / sw;
        double sy = (double) (H) / sh;
        double scale = Math.min(sx, sy);

        int tw = Math.max(1, (int)Math.round(sw*scale));
        int th = Math.max(1, (int)Math.round(sh*scale));

        BufferedImage scaled = new BufferedImage(tw, th, BufferedImage.TYPE_INT_RGB);
        Graphics2D g0 = scaled.createGraphics();
        g0.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
        g0.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g0.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        AffineTransform at = AffineTransform.getScaleInstance(scale, scale);
        g0.drawRenderedImage(src, at);
        g0.dispose();

        BufferedImage canvas = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = canvas.createGraphics();
        g.setColor(bg);
        g.fillRect(0,0,W,H);
        int ox = (W - tw)/2;
        int oy = (H - th)/2;
        g.drawImage(scaled, ox, oy, null);
        g.dispose();
        return canvas;
    }

    /* ========== 统计 & 清理 ========== */
    static void statInc(Map<String,Integer> map, String k){
        map.merge(k, 1, Integer::sum);
    }

    static void printStat(Map<String,Integer> m, int total){
        List<String> keys = new ArrayList<>(m.keySet());
        Collections.sort(keys);
        for (String k : keys){
            int v = m.getOrDefault(k,0);
            double pct = total>0? (v*100.0/total) : 0;
            System.out.printf("  %-12s : %7d  (%5.1f%%)\n", k, v, pct);
        }
    }

    static void deleteDirRecursive(Path p) throws IOException {
        if (p==null || !Files.exists(p)) return;
        Files.walkFileTree(p, new SimpleFileVisitor<Path>(){
            @Override public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                Files.deleteIfExists(file); return FileVisitResult.CONTINUE;
            }
            @Override public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException {
                Files.deleteIfExists(dir); return FileVisitResult.CONTINUE;
            }
        });
    }
}
