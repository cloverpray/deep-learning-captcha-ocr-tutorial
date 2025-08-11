// -*- coding: utf-8 -*-
// CalcAllStylesRunner.java  (TEXT ONLY, with SMART mode - FIXED TYPING)
//
// 编译：
//   javac -encoding UTF-8 -cp ".;CalculateCaptcha-1.1.jar" CalcAllStylesRunner.java
//
// 查看类型列表：
//   java -cp ".;CalculateCaptcha-1.1.jar" CalcAllStylesRunner list
//
// 传统生成：
//   java -cp ".;CalculateCaptcha-1.1.jar" CalcAllStylesRunner gen text:water:wide:nonoise out_dir 2000 160 60 4
//   java -cp ".;CalculateCaptcha-1.1.jar" CalcAllStylesRunner all out_root 500 160 60 4
//
// SMART 自动分布（推荐）：
//   java -cp ".;CalculateCaptcha-1.1.jar" CalcAllStylesRunner smart <outDir> <total> \
//     [--sizes "160x60:40,180x64:20,200x64:15,256x64:10,120x40:10,96x32:5"] \
//     [--len "4:50,5:25,6:25" | --fixed-len 4] \
//     [--fonts "Arial,Microsoft YaHei,SimSun,Noto Sans CJK SC"] \
//     [--charset "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"] \
//     [--palette "#000000@40,#1C5BD5@15,#228B22@10,#8B0000@8,#4B0082@8,#D2691E@8,#008080@6,#2F4F4F@5"] \
//     [--threads 8]
//
import com.google.code.kaptcha.impl.DefaultKaptcha;
import com.google.code.kaptcha.text.TextProducer;
import com.google.code.kaptcha.util.Config;
import com.google.code.kaptcha.util.Configurable;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class CalcAllStylesRunner {

    /* ========== 文本 Producer（遵从 config 的 char.string & char.length） ========== */
    public static class MixedTextProducer extends Configurable implements TextProducer {
        private final Random rnd = new Random();
        @Override public String getText() {
            String src = null;
            try {
                char[] arr = getConfig().getTextProducerCharString(); // 新版返回 char[]
                if (arr!=null && arr.length>0) src = new String(arr);
            }catch(Throwable ignore){
                try { // 兼容旧版
                    Object val = Config.class.getMethod("getTextProducerCharString").invoke(getConfig());
                    if (val instanceof char[]) src = new String((char[])val);
                    else if (val!=null) src = String.valueOf(val);
                }catch(Throwable e){ /* ignore */ }
            }
            if (src==null || src.isEmpty()){
                src = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
            }
            int L = 4;
            try { L = Math.max(1, getConfig().getTextProducerCharLength()); } catch(Throwable ignore){}
            StringBuilder sb = new StringBuilder(L);
            for (int i=0;i<L;i++) sb.append(src.charAt(rnd.nextInt(src.length())));
            return sb.toString();
        }
    }

    /* ========== 风格定义（仅文本） ========== */
    enum Dist { WATER, SHADOW, FISHEYE }
    enum Space { NORMAL, WIDE, COMPACT }
    enum Noise { DEFAULT, NONOISE }
    static class Kind {
        final Dist dist; final Space space; final Noise noise;
        Kind(Dist d, Space s, Noise n){ dist=d; space=s; noise=n; }
        String code(){
            return "text:" + dist.name().toLowerCase(Locale.ROOT)
                    + ":" + space.name().toLowerCase(Locale.ROOT)
                    + ":" + (noise==Noise.NONOISE?"nonoise":"noise");
        }
    }
    static final java.util.List<Kind> ALL_TEXT_KINDS = new ArrayList<>();
    static {
        for (Dist d : Dist.values())
            for (Space s : Space.values())
                for (Noise n : Noise.values())
                    ALL_TEXT_KINDS.add(new Kind(d,s,n)); // 3*3*2 = 18
    }

    /* ========== 默认字体/颜色 ========== */
    static final String[] SAFE_FONTS = new String[]{
            "Arial","Helvetica","Microsoft YaHei","SimSun","SimHei","Noto Sans CJK SC","Tahoma","Verdana"
    };
    static final String DEFAULT_PALETTE =
            "#000000@40,#1C5BD5@15,#228B22@10,#8B0000@8,#4B0082@8,#D2691E@8,#008080@6,#2F4F4F@5";
    static final String DEFAULT_SIZES =
            "160x60:40,180x64:20,200x64:15,256x64:10,120x40:10,96x32:5";
    static final String DEFAULT_LEN_DIST = "4:50,5:25,6:25";

    /* ========== 工具方法 & 参数解析 ========== */
    static String opt(String[] a, String key, String def){
        for (int i=0;i<a.length;i++){
            if (key.equals(a[i]) && i+1<a.length) return a[i+1];
        }
        return def;
    }
    static boolean hasFlag(String[] a, String key){
        for (String s : a) if (key.equals(s)) return true;
        return false;
    }
    static Map<String,String> parseOpts(String[] a, int from){
        Map<String,String> m = new HashMap<>();
        for (int i=from;i<a.length;i++){
            String k = a[i];
            if (k.startsWith("--")){
                String key = k.substring(2);
                String val = (i+1<a.length && !a[i+1].startsWith("--")) ? a[++i] : "1";
                m.put(key, val);
            }
        }
        return m;
    }
    static String[] parseFonts(String s){
        if (s==null || s.trim().isEmpty()) return SAFE_FONTS;
        String[] sp = s.split(",");
        java.util.List<String> out = new ArrayList<>();
        for (String x : sp){ String t=x.trim(); if (!t.isEmpty()) out.add(t); }
        if (out.isEmpty()) return SAFE_FONTS;
        return out.toArray(new String[0]);
    }
    static String safeName(String s){
        if (s==null) return "out";
        String r = s.replaceAll("[\\\\/:*?\"<>|\\s]+", "_").replaceAll("_+","_");
        if (r.startsWith("_")) r = r.substring(1);
        if (r.isEmpty()) r = "out";
        return r;
    }
    static String uuid(){ return java.util.UUID.randomUUID().toString().replace("-", ""); }

    /* ========== 颜色权重解析/采样（强类型） ========== */
    static class WColor { String rgb; double w; WColor(String rgb,double w){ this.rgb=rgb; this.w=w; } }
    static java.util.List<WColor> parsePalette(String s){
        java.util.List<WColor> list = new ArrayList<>();
        if (s==null || s.trim().isEmpty()) s = DEFAULT_PALETTE;
        String[] items = s.split(",");
        for (String it : items){
            it = it.trim(); if (it.isEmpty()) continue;
            String[] kv = it.split("@");
            String hexOrRGB = kv[0].trim();
            double w = (kv.length>1)? parseDoubleSafe(kv[1],1.0) : 1.0;
            list.add(new WColor(toRgbTriplet(hexOrRGB), Math.max(0,w)));
        }
        if (list.isEmpty()) list.add(new WColor("0,0,0",1));
        double sum = 0; for (WColor c : list) sum += c.w;
        if (sum<=0){ list.get(0).w = 1; sum=1; }
        for (WColor c : list) c.w /= sum;
        return list;
    }
    static String toRgbTriplet(String hex){
        hex = hex.trim();
        if (!hex.startsWith("#")) return hex; // 已是 "r,g,b"
        try{
            int r,g,b;
            if (hex.length()==7){
                r = Integer.parseInt(hex.substring(1,3),16);
                g = Integer.parseInt(hex.substring(3,5),16);
                b = Integer.parseInt(hex.substring(5,7),16);
            }else if (hex.length()==4){
                r = Integer.parseInt(hex.substring(1,2),16)*17;
                g = Integer.parseInt(hex.substring(2,3),16)*17;
                b = Integer.parseInt(hex.substring(3,4),16)*17;
            }else return "0,0,0";
            return r + "," + g + "," + b;
        }catch(Exception e){ return "0,0,0"; }
    }
    static double parseDoubleSafe(String s, double def){ try{ return Double.parseDouble(s.trim()); }catch(Exception e){ return def; } }
    static WColor sampleColor(java.util.List<WColor> palette, Random rnd){
        double r = rnd.nextDouble(), acc=0;
        for (WColor c : palette){
            acc += c.w;
            if (r <= acc) return c;
        }
        return palette.get(palette.size()-1);
    }

    /* ========== 尺寸/长度分布解析/采样（强类型） ========== */
    static class SizeBucket { int W,H; double w; SizeBucket(int W,int H,double w){ this.W=W; this.H=H; this.w=w; } }
    static java.util.List<SizeBucket> parseSizes(String sizes){
        if (sizes==null || sizes.trim().isEmpty()) sizes = DEFAULT_SIZES;
        java.util.List<SizeBucket> out = new ArrayList<>();
        String[] items = sizes.split(",");
        for (String it : items){
            it = it.trim(); if (it.isEmpty()) continue;
            String[] kv = it.split(":");
            String wh = kv[0].trim();
            double w = (kv.length>1)? parseDoubleSafe(kv[1],1.0) : 1.0;
            String[] whp = wh.toLowerCase(Locale.ROOT).split("x");
            int W = Integer.parseInt(whp[0].trim());
            int H = Integer.parseInt(whp[1].trim());
            out.add(new SizeBucket(W,H,Math.max(0,w)));
        }
        if (out.isEmpty()) out.add(new SizeBucket(160,60,1));
        double sum = 0; for (SizeBucket b : out) sum += b.w;
        if (sum<=0){ out.get(0).w = 1; sum=1; }
        for (SizeBucket b : out) b.w /= sum;
        return out;
    }
    static class LenBucket { int L; double w; LenBucket(int L,double w){ this.L=L; this.w=w; } }
    static java.util.List<LenBucket> parseLenDist(String s){
        if (s==null || s.trim().isEmpty()) s = DEFAULT_LEN_DIST;
        java.util.List<LenBucket> out = new ArrayList<>();
        String[] items = s.split(",");
        for (String it : items){
            it = it.trim(); if (it.isEmpty()) continue;
            String[] kv = it.split(":");
            int L = Integer.parseInt(kv[0].trim());
            double w = (kv.length>1)? parseDoubleSafe(kv[1],1.0) : 1.0;
            out.add(new LenBucket(L, Math.max(0,w)));
        }
        if (out.isEmpty()){ out.add(new LenBucket(4,1)); out.add(new LenBucket(5,1)); out.add(new LenBucket(6,1)); }
        double sum = 0; for (LenBucket b : out) sum += b.w;
        if (sum<=0){ for (LenBucket b : out) b.w = 1.0/out.size(); }
        else for (LenBucket b : out) b.w /= sum;
        return out;
    }
    static SizeBucket sampleSize(java.util.List<SizeBucket> buckets, Random rnd){
        double r = rnd.nextDouble(), acc=0;
        for (SizeBucket b : buckets){
            acc += b.w;
            if (r <= acc) return b;
        }
        return buckets.get(buckets.size()-1);
    }
    static LenBucket sampleLen(java.util.List<LenBucket> buckets, Random rnd){
        double r = rnd.nextDouble(), acc=0;
        for (LenBucket b : buckets){
            acc += b.w;
            if (r <= acc) return b;
        }
        return buckets.get(buckets.size()-1);
    }

    /* ========== Kaptcha 属性构建 ========== */
    static String mapDist(Dist d){
        switch (d){
            case WATER:   return "com.google.code.kaptcha.impl.WaterRipple";
            case SHADOW:  return "com.google.code.kaptcha.impl.ShadowGimpy";
            case FISHEYE: return "com.google.code.kaptcha.impl.FishEyeGimpy";
        }
        return "com.google.code.kaptcha.impl.WaterRipple";
    }
    static Properties buildPropsText(int W, int H, int len, Kind k,
                                     String[] fontNames, String charsetNoSpace){
        Properties p = new Properties();
        p.setProperty("kaptcha.image.width",  String.valueOf(W));
        p.setProperty("kaptcha.image.height", String.valueOf(H));
        p.setProperty("kaptcha.background.impl", "com.google.code.kaptcha.impl.DefaultBackground");
        p.setProperty("kaptcha.background.clear.from", "255,255,255");
        p.setProperty("kaptcha.background.clear.to",   "233,238,243");
        p.setProperty("kaptcha.border", "no");

        p.setProperty("kaptcha.obscurificator.impl", mapDist(k.dist));
        if (k.noise==Noise.NONOISE){
            p.setProperty("kaptcha.noise.impl", "com.google.code.kaptcha.impl.NoNoise");
        }else{
            p.setProperty("kaptcha.noise.impl", "com.google.code.kaptcha.impl.DefaultNoise");
            p.setProperty("kaptcha.noise.color", "black");
        }

        p.setProperty("kaptcha.textproducer.impl", "CalcAllStylesRunner$MixedTextProducer");
        p.setProperty("kaptcha.textproducer.char.string", charsetNoSpace);
        p.setProperty("kaptcha.textproducer.char.length", String.valueOf(Math.max(1,len)));

        int space = (k.space==Space.WIDE?6 : k.space==Space.COMPACT?1 : 3);
        p.setProperty("kaptcha.textproducer.char.space", String.valueOf(space));
        p.setProperty("kaptcha.textproducer.font.names", String.join(",", fontNames));
        int fontSize = Math.max(16, Math.round(H * 0.72f));
        p.setProperty("kaptcha.textproducer.font.size", String.valueOf(fontSize));
        // 颜色在生成时动态注入：kaptcha.textproducer.font.color
        return p;
    }

    /* ========== 主入口 ========== */
    public static void main(String[] args) {
        if (args.length==0 || "help".equalsIgnoreCase(args[0])) { printHelp(); return; }
        try{
            switch (args[0].toLowerCase(Locale.ROOT)){
                case "list": doList(); break;
                case "gen":  doGen(Arrays.copyOfRange(args,1,args.length)); break;
                case "all":  doAll(Arrays.copyOfRange(args,1,args.length)); break;
                case "smart":doSmart(Arrays.copyOfRange(args,1,args.length)); break;
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
        System.out.println("  list");
        System.out.println("  gen  <type> <out> <count> <W> <H> <len> [--fonts ...] [--charset ...] [--palette ...]");
        System.out.println("  all  <outRoot> <perType> <W> <H> <len> [--fonts ...] [--charset ...] [--palette ...]");
        System.out.println("  smart <outDir> <total> [--sizes \""+DEFAULT_SIZES+"\"] [--len \""+DEFAULT_LEN_DIST+"\"] [--fixed-len N]");
        System.out.println("        [--fonts \"Arial,Microsoft YaHei,SimSun\"] [--charset \"0-9A-Za-z\"] [--palette \""+DEFAULT_PALETTE+"\"] [--threads 8]");
        System.out.println();
        System.out.println("type = text:<dist>:<space>:<noise>  例如 text:water:wide:nonoise");
        System.out.println("  dist : water|shadow|fisheye");
        System.out.println("  space: normal|wide|compact");
        System.out.println("  noise: noise|nonoise");
    }

    static void doList(){
        System.out.println("可用类型（文本，共 " + ALL_TEXT_KINDS.size() + " 种组合）：");
        for (Kind k : ALL_TEXT_KINDS) System.out.println("  - " + k.code());
    }

    /* 传统 gen/all */
    static Kind parseKind(String s){
        String[] a = s.split(":");
        if (a.length!=4 || !"text".equalsIgnoreCase(a[0]))
            throw new IllegalArgumentException("type 必须是 text:<dist>:<space>:<noise>");
        Dist d = Dist.valueOf(a[1].toUpperCase(Locale.ROOT));
        Space sp = Space.valueOf(a[2].toUpperCase(Locale.ROOT));
        Noise n = "nonoise".equalsIgnoreCase(a[3]) ? Noise.NONOISE : Noise.DEFAULT;
        return new Kind(d,sp,n);
    }
    static void doGen(String[] argv) throws Exception {
        if (argv.length < 6){
            System.out.println("参数不足。用法：gen <type> <out> <count> <W> <H> <len> [--fonts ...] [--charset ...] [--palette ...]");
            return;
        }
        Kind kind = parseKind(argv[0]);
        String outName = safeName(argv[1]);
        int count = Integer.parseInt(argv[2]);
        int W = Integer.parseInt(argv[3]);
        int H = Integer.parseInt(argv[4]);
        int len = Integer.parseInt(argv[5]);

        Map<String,String> opt = parseOpts(argv, 6);
        String fontsArg   = opt.getOrDefault("fonts", String.join(",", SAFE_FONTS));
        String charsetArg = opt.getOrDefault("charset", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
        String paletteArg = opt.getOrDefault("palette", DEFAULT_PALETTE);
        java.util.List<WColor> palette = parsePalette(paletteArg);
        String[] fonts = parseFonts(fontsArg);
        String charset = charsetArg.replace(" ", "");

        Path outDir  = Paths.get("images").resolve(outName);
        Path outImgs = outDir.resolve("images");
        Path outLbl  = outDir.resolve("labels.txt");
        Files.createDirectories(outImgs);
        System.out.printf(">> %s -> %s  (count=%d, %dx%d, len=%d)\n", kind.code(), outDir, count, W, H, len);

        Random rnd = new Random();
        try (BufferedWriter bw = Files.newBufferedWriter(outLbl, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            for (int i=0;i<count;i++){
                Properties p = buildPropsText(W,H,len,kind,fonts,charset);
                WColor col = sampleColor(palette, rnd);
                p.setProperty("kaptcha.textproducer.font.color", col.rgb);

                DefaultKaptcha kk = new DefaultKaptcha(); kk.setConfig(new Config(p));
                String text = kk.createText();
                BufferedImage img = kk.createImage(text);

                String fn = uuid()+".png";
                ImageIO.write(img, "png", outImgs.resolve(fn).toFile());
                bw.write(fn); bw.write('\t'); bw.write(text); bw.newLine();

                if ((i+1)%200==0 || i+1==count) System.out.printf("\r.. %d/%d", i+1, count);
            }
        }
        System.out.println("\n✅ 完成：" + outDir.toAbsolutePath());
    }
    static void doAll(String[] argv) throws Exception {
        if (argv.length < 5){
            System.out.println("参数不足。用法：all <outRoot> <perType> <W> <H> <len> [--fonts ...] [--charset ...] [--palette ...]");
            return;
        }
        String rootName = safeName(argv[0]);
        int per = Integer.parseInt(argv[1]);
        int W   = Integer.parseInt(argv[2]);
        int H   = Integer.parseInt(argv[3]);
        int L   = Integer.parseInt(argv[4]);

        Map<String,String> opt = parseOpts(argv, 5);
        String fontsArg   = opt.getOrDefault("fonts", String.join(",", SAFE_FONTS));
        String charsetArg = opt.getOrDefault("charset", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
        String paletteArg = opt.getOrDefault("palette", DEFAULT_PALETTE);

        String[] fonts = parseFonts(fontsArg);
        String charset = charsetArg.replace(" ", "");
        Random rnd = new Random();
        java.util.List<WColor> palette = parsePalette(paletteArg);

        for (Kind k : ALL_TEXT_KINDS){
            String outName = safeName(rootName + "_" + k.code());
            Path outDir  = Paths.get("images").resolve(outName);
            Path outImgs = outDir.resolve("images");
            Path outLbl  = outDir.resolve("labels.txt");
            Files.createDirectories(outImgs);

            System.out.printf(">> %s -> %s  (count=%d, %dx%d, len=%d)\n", k.code(), outDir, per, W, H, L);
            Properties base = buildPropsText(W,H,L,k,fonts,charset);

            try (BufferedWriter bw = Files.newBufferedWriter(outLbl, StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
                for (int i=0;i<per;i++){
                    Properties p = new Properties(); p.putAll(base);
                    WColor col = sampleColor(palette, rnd);
                    p.setProperty("kaptcha.textproducer.font.color", col.rgb);

                    DefaultKaptcha kk = new DefaultKaptcha(); kk.setConfig(new Config(p));
                    String text = kk.createText();
                    BufferedImage img = kk.createImage(text);

                    String fn = uuid()+".png";
                    ImageIO.write(img, "png", outImgs.resolve(fn).toFile());
                    bw.write(fn); bw.write('\t'); bw.write(text); bw.newLine();

                    if ((i+1)%200==0 || i+1==per) System.out.printf("\r.. %d/%d", i+1, per);
                }
            }
            System.out.println("\n   完成：" + outDir.toAbsolutePath());
        }
        System.out.println("✅ ALL DONE.");
    }

    /* ========== SMART 自动分布 ========== */
    static void doSmart(String[] argv) throws Exception {
        if (argv.length < 2){
            System.out.println("参数不足。用法：smart <outDir> <total> [--sizes ...] [--len ... | --fixed-len N] [--fonts ...] [--charset ...] [--palette ...] [--threads N]");
            return;
        }
        String outName = safeName(argv[0]);
        int total = Integer.parseInt(argv[1]);

        Map<String,String> opt = parseOpts(argv, 2);
        String sizesArg   = opt.getOrDefault("sizes",   DEFAULT_SIZES);
        String lenArg     = opt.getOrDefault("len",     DEFAULT_LEN_DIST);
        int fixedLen      = Integer.parseInt(opt.getOrDefault("fixed-len", "0"));
        String fontsArg   = opt.getOrDefault("fonts",   String.join(",", SAFE_FONTS));
        String charsetArg = opt.getOrDefault("charset", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
        String paletteArg = opt.getOrDefault("palette", DEFAULT_PALETTE);
        int threads       = Integer.parseInt(opt.getOrDefault("threads", String.valueOf(Math.max(1, Runtime.getRuntime().availableProcessors()))));

        java.util.List<SizeBucket> sizeBuckets = parseSizes(sizesArg);
        java.util.List<LenBucket>  lenBuckets  = (fixedLen>0 ? Arrays.asList(new LenBucket(fixedLen,1)) : parseLenDist(lenArg));
        java.util.List<WColor>     palette     = parsePalette(paletteArg);
        String[] fonts = parseFonts(fontsArg);
        String charset = charsetArg.replace(" ", "");

        Path outDir  = Paths.get("images").resolve(outName);
        Path outImgs = outDir.resolve("images");
        Path outLbl  = outDir.resolve("labels.txt");
        Files.createDirectories(outImgs);
        System.out.printf(">> SMART text-mix -> %s  (total=%d)\n", outDir, total);
        System.out.println("   sizes   = " + sizesArg);
        System.out.println("   len     = " + (fixedLen>0? ("fixed="+fixedLen) : lenArg));
        System.out.println("   threads = " + threads);

        Random rnd = new Random();
        AtomicInteger counter = new AtomicInteger(0);

        // 统计
        Map<String,Integer> statSize  = new ConcurrentHashMap<>();
        Map<String,Integer> statKind  = new ConcurrentHashMap<>();
        Map<String,Integer> statLen   = new ConcurrentHashMap<>();

        ExecutorService pool = Executors.newFixedThreadPool(threads);
        final Object labelLock = new Object();

        try (BufferedWriter bw = Files.newBufferedWriter(outLbl, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {

            for (int t=0;t<total;t++){
                pool.submit(() -> {
                    try{
                        // 采样：风格均匀、尺寸/长度/颜色按权重
                        Kind kind = ALL_TEXT_KINDS.get(rnd.nextInt(ALL_TEXT_KINDS.size()));
                        SizeBucket sb = sampleSize(sizeBuckets, rnd);
                        LenBucket  lb = sampleLen(lenBuckets,  rnd);
                        WColor     wc = sampleColor(palette,   rnd);

                        Properties p = buildPropsText(sb.W, sb.H, lb.L, kind, fonts, charset);
                        p.setProperty("kaptcha.textproducer.font.color", wc.rgb);

                        DefaultKaptcha kk = new DefaultKaptcha(); kk.setConfig(new Config(p));
                        String text = kk.createText();
                        BufferedImage img = kk.createImage(text);

                        int idx = counter.incrementAndGet();
                        String fn = String.format("%08d.png", idx);
                        ImageIO.write(img, "png", outImgs.resolve(fn).toFile());

                        synchronized (labelLock){
                            bw.write(fn); bw.write('\t'); bw.write(text); bw.newLine();
                        }

                        // 统计
                        statInc(statSize, sb.W+"x"+sb.H);
                        statInc(statKind, kind.code());
                        statInc(statLen,  String.valueOf(lb.L));

                        if (idx % 200 == 0 || idx==total) {
                            System.out.printf("\r.. %d/%d", idx, total);
                        }
                    }catch(Exception e){
                        e.printStackTrace();
                    }
                });
            }
            pool.shutdown();
            pool.awaitTermination(3, TimeUnit.DAYS);
        }

        System.out.println("\n✅ SMART 完成：" + outDir.toAbsolutePath());
        // 打印统计
        System.out.println("== 尺寸分布 ==");
        printStat(statSize, total);
        System.out.println("\n== 风格分布（18种） ==");
        printStat(statKind, total);
        System.out.println("\n== 文本长度分布 ==");
        printStat(statLen, total);
    }

    static void statInc(Map<String,Integer> map, String k){
        map.merge(k, 1, Integer::sum);
    }
    static void printStat(Map<String,Integer> m, int total){
        java.util.List<String> keys = new ArrayList<>(m.keySet());
        Collections.sort(keys);
        for (String k : keys){
            int v = m.getOrDefault(k,0);
            double pct = total>0 ? (v*100.0/total) : 0;
            System.out.printf("  %-22s : %7d  (%5.1f%%)\n", k, v, pct);
        }
    }
}
