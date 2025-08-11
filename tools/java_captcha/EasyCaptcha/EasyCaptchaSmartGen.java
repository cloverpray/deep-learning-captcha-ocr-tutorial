// -*- coding: utf-8 -*-
// 科学分布版验证码生成器（加速 & 算术操作符纯随机版 + 进度条 + 统计落盘）
//
// 依赖：easy-captcha-1.6.2-RELEASE.jar
// JDK15+ 跑算术需：nashorn-core-15.4.jar、asm-9.6.jar、asm-commons-9.6.jar、asm-util-9.6.jar、asm-tree-9.6.jar、asm-analysis-9.6.jar
//
// 编译：
//   javac -encoding UTF-8 -cp "easy-captcha-1.6.2-RELEASE.jar" EasyCaptchaSmartGen.java
//
// 示例（默认分布 + 内置字体 + 多线程）：
//   java -cp ".;easy-captcha-1.6.2-RELEASE.jar;nashorn-core-15.4.jar;asm-9.6.jar;asm-commons-9.6.jar;asm-util-9.6.jar;asm-tree-9.6.jar;asm-analysis-9.6.jar" ^
//     EasyCaptchaSmartGen smart out_smart 10000 --fonts builtin --threads 8
//
// 功能要点：
// - 文本长度分布：严格 4:50% / 5:25% / 6:25%（配额+洗牌）
// - 算术操作符：纯随机（不配额、不重试）
// - 多线程：--threads N（默认=CPU核数）
// - 输出：./images/<outName>/images/*.png|gif + labels.txt
// - 统计：终端打印 + 写入 stats.json / stats.txt
// - 进度：显示已完成/总数、百分比、速度与 ETA
//
// 提示：GIF 只有当你在 --text-mix 中包含 gif 类别时才会生成

import com.wf.captcha.*;
import com.wf.captcha.base.Captcha;

import java.awt.Font;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class EasyCaptchaSmartGen {

    public static void main(String[] args) throws Exception {
        if (args.length < 3 || "help".equalsIgnoreCase(args[0])) { printHelp(); return; }
        if (!"smart".equalsIgnoreCase(args[0])) { System.out.println("仅支持命令：smart（查看帮助：help）"); return; }

        final String outName = args[1];
        final int total = parseInt(args[2], 10000);

        // 分布（支持命令行覆盖）
        LinkedHashMap<String,Integer> sizePlan = parseSizesPlan(getArg(args, "--sizes",
            "160x60:40,180x64:20,200x64:15,256x64:10,120x40:10,96x32:5"));
        LinkedHashMap<String,Integer> taskPlan = parseKVPlan(getArg(args, "--task", "text:70,arith:30"));
        LinkedHashMap<String,Integer> textMix  = parseKVPlan(getArg(args, "--text-mix",
            "spec:50,spec_numupper:25,spec_numlower:10,spec_upper:10,spec_lower:5"));
        LinkedHashMap<String,Integer> lenPlan  = parseKVPlan(getArg(args, "--len", "4:50,5:25,6:25"));
        LinkedHashMap<String,Integer> arithLen = parseKVPlan(getArg(args, "--arith-len", "2:80,3:20"));

        final int threads = Math.max(1, parseInt(getArg(args, "--threads",
                String.valueOf(Runtime.getRuntime().availableProcessors())), 1));
        FontPlan fontPlan = parseFontPlan(args);

        // 输出目录
        Path ROOT = Paths.get("images");
        Path outDir = ROOT.resolve(outName);
        Path imgDir = outDir.resolve("images");
        Path lbl    = outDir.resolve("labels.txt");
        Files.createDirectories(imgDir);

        // 预检测算术依赖（仅当需要算术）
        int wantArith = allocFromPercent(total, taskPlan.getOrDefault("arith", 30));
        ensureArithDepsIfNeeded(wantArith);

        // 预构建严格配额列表
        List<String> sizeList = buildListFromPercent(total, sizePlan);
        List<String> taskList = buildListFromPercent(total, taskPlan);
        int numText  = countOf(taskList, "text");
        int numArith = countOf(taskList, "arith");
        List<String> textTypeList = buildListFromPercent(numText, textMix);
        List<Integer> textLenList = buildLenList(numText, lenPlan);     // 4/5/6
        List<Integer> arithLenList= buildLenList(numArith, arithLen);   // 2/3

        // 预解析字体池（算术兜底）以减少每张图检测
        prepareFontPools(fontPlan);

        // 组装样本计划（严格配额）
        List<Spec> specs = new ArrayList<>(total);
        int idxText=0, idxArith=0;
        for (int i=0;i<total;i++){
            String sz = sizeList.get(i);
            int W = Integer.parseInt(sz.split("x")[0]);
            int H = Integer.parseInt(sz.split("x")[1]);
            String task = taskList.get(i);
            if ("text".equals(task)){
                String tp = textTypeList.get(idxText);
                int L = textLenList.get(idxText);
                idxText++;
                specs.add(Spec.text(W,H,tp,L));
            }else{
                int aL = arithLenList.get(idxArith);
                idxArith++;
                specs.add(Spec.arith(W,H,aL));
            }
        }

        // 统计（计划分布）
        Map<String,Integer> sizeCnt = new LinkedHashMap<>();
        Map<String,Integer> taskCnt = new LinkedHashMap<>();
        Map<String,Integer> typeCnt = new LinkedHashMap<>();
        Map<Integer,Integer> tlenCnt= new LinkedHashMap<>();
        Map<Integer,Integer> alenCnt= new LinkedHashMap<>();
        for (Spec s : specs){
            inc(sizeCnt, s.W + "x" + s.H);
            inc(taskCnt, s.isText() ? "text":"arith");
            if (s.isText()) { inc(typeCnt, s.textType); inc(tlenCnt, s.textLen); }
            else { inc(alenCnt, s.arithLen); }
        }

        // 算术操作符实际分布（纯随机统计）
        ConcurrentHashMap<String,AtomicInteger> opCnt = new ConcurrentHashMap<>();

        // 多线程执行 + 进度条
        System.out.printf(">> smart -> %s  (total=%d, threads=%d)\n", outDir, total, threads);
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        ExecutorCompletionService<LineOut> ecs = new ExecutorCompletionService<>(pool);

        for (int i=0;i<total;i++){
            final int idx = i;
            ecs.submit(() -> {
                Spec s = specs.get(idx);
                String fn;
                String label;
                if (s.isText()){
                    // 纯文本
                    if (s.textType.startsWith("gif")) {
                        GifCaptcha c = new GifCaptcha(s.W, s.H, s.textLen);
                        c.setCharType(mapCharType(s.textType));
                        applyFontIfAny(c, fontPlan, s.H, idx);
                        label = c.text();
                        fn = uuid()+".gif";
                        try (OutputStream os = new BufferedOutputStream(Files.newOutputStream(imgDir.resolve(fn)))) { c.out(os); }
                    } else if ("cn".equals(s.textType) || "cn_gif".equals(s.textType)) {
                        boolean gif = "cn_gif".equals(s.textType);
                        Captcha c = gif ? new ChineseGifCaptcha(s.W,s.H,s.textLen) : new ChineseCaptcha(s.W,s.H,s.textLen);
                        applyFontIfAny(c, fontPlan, s.H, idx);
                        label = c.text();
                        fn = uuid() + (gif?".gif":".png");
                        try (OutputStream os = new BufferedOutputStream(Files.newOutputStream(imgDir.resolve(fn)))) { c.out(os); }
                    } else {
                        SpecCaptcha c = new SpecCaptcha(s.W, s.H, s.textLen);
                        c.setCharType(mapCharType(s.textType));
                        applyFontIfAny(c, fontPlan, s.H, idx);
                        label = c.text();
                        fn = uuid()+".png";
                        try (OutputStream os = new BufferedOutputStream(Files.newOutputStream(imgDir.resolve(fn)))) { c.out(os); }
                    }
                    return new LineOut(fn, label, null);
                } else {
                    // 算术（操作符纯随机：不配额、不重试）
                    ArithmeticCaptcha c = new ArithmeticCaptcha(s.W, s.H);
                    c.setLen(s.arithLen);
                    applyFontForArith(c, fontPlan, s.H, idx);
                    String expr = c.getArithmeticString(); // 例如 “8+6=？”
                    String op = extractOp(expr);           // 统计用
                    if (op!=null) opCnt.computeIfAbsent(op, k->new AtomicInteger()).incrementAndGet();
                    fn = uuid()+".png";
                    try (OutputStream os = new BufferedOutputStream(Files.newOutputStream(imgDir.resolve(fn)))) { c.out(os); }
                    return new LineOut(fn, expr, op);
                }
            });
        }
        pool.shutdown();

        // 进度条打印器
        ProgressPrinter prog = new ProgressPrinter(total);
        List<LineOut> results = new ArrayList<>(total);

        try (BufferedWriter bw = Files.newBufferedWriter(lbl, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            for (int i=0;i<total;i++){
                Future<LineOut> f = ecs.take(); // 按完成顺序取
                LineOut lo;
                try { lo = f.get(); }
                catch (ExecutionException ee){
                    prog.finishWithError();
                    throw new RuntimeException("生成任务失败: " + ee.getCause(), ee.getCause());
                }
                results.add(lo);
                bw.write(lo.filename + "\t" + lo.label); bw.newLine();
                prog.step(); // 更新进度
            }
        }
        prog.done(); // 完成换行

        // 终端统计
        System.out.println();
        printTable("尺寸分布", sizeCnt, total);
        printTable("任务分布", taskCnt, total);
        printTable("文本类型分布（text-mix）", typeCnt, sum(typeCnt));
        printTableInt("文本长度分布（len）", tlenCnt, sum(tlenCnt));
        printTableInt("算术参与数分布（arith-len）", alenCnt, sum(alenCnt));
        // 实际算术操作符分布（纯随机统计结果）
        LinkedHashMap<String,Integer> opCntFinal = new LinkedHashMap<>();
        for (Map.Entry<String,AtomicInteger> e : opCnt.entrySet()) opCntFinal.put(e.getKey(), e.getValue().get());
        printTable("算术操作符分布（实际）", opCntFinal, sum(opCntFinal));

        // 统计落盘
        writeStatsJson(outDir, args, sizePlan, taskPlan, textMix, lenPlan, arithLen,
                sizeCnt, taskCnt, typeCnt, tlenCnt, alenCnt, opCntFinal, threads, total);
        writeStatsTxt(outDir, sizeCnt, taskCnt, typeCnt, tlenCnt, alenCnt, opCntFinal, total);

        System.out.println("\n✅ 生成完毕 -> " + outDir.toAbsolutePath());
        System.out.println("📄 统计已写入： " + outDir.resolve("stats.json") + "  和  " + outDir.resolve("stats.txt"));
    }

    // -------------------- 帮助 --------------------
    static void printHelp(){
        System.out.println("用法:");
        System.out.println("  java -cp \".;easy-captcha-1.6.2-RELEASE.jar[;nashorn;asm...]\" EasyCaptchaSmartGen smart <outName> <total> [选项]");
        System.out.println("选项:");
        System.out.println("  --sizes       尺寸分布，默认 160x60:40,180x64:20,200x64:15,256x64:10,120x40:10,96x32:5");
        System.out.println("  --task        任务分布 text:70,arith:30");
        System.out.println("  --text-mix    文本类型分布 spec:50,spec_numupper:25,spec_numlower:10,spec_upper:10,spec_lower:5");
        System.out.println("  --len         文本长度分布 4:50,5:25,6:25");
        System.out.println("  --arith-len   算术参与数分布 2:80,3:20");
        System.out.println("  --fonts       builtin 或 逗号分隔的系统字体名，如 宋体,Consolas,Microsoft YaHei");
        System.out.println("  --threads     并发线程数，默认=CPU核数");
        System.out.println();
        System.out.println("输出：images/<outName>/images/*.png|gif  和  images/<outName>/labels.txt");
        System.out.println("统计：images/<outName>/stats.json  与  stats.txt");
    }

    // -------------------- 数据结构 --------------------
    static class Spec {
        final int W,H;
        final boolean isText;
        final String textType; // for text
        final int textLen;     // for text
        final int arithLen;    // for arith
        private Spec(int W,int H, boolean isText, String textType, int textLen, int arithLen){
            this.W=W; this.H=H; this.isText=isText; this.textType=textType; this.textLen=textLen; this.arithLen=arithLen;
        }
        static Spec text(int W,int H,String tp,int L){ return new Spec(W,H,true,tp,L,0); }
        static Spec arith(int W,int H,int aL){ return new Spec(W,H,false,null,0,aL); }
        boolean isText(){ return isText; }
    }
    static class LineOut {
        final String filename, label, op;
        LineOut(String f,String l,String o){ filename=f; label=l; op=o; }
    }

    // -------------------- 进度条 --------------------
    static class ProgressPrinter {
        final int total;
        final long start;
        final int barLen = 36;
        volatile int done = 0;
        volatile long lastPrint = 0L;
        ProgressPrinter(int total){
            this.total = total;
            this.start = System.nanoTime();
            print(0);
        }
        void step(){
            int cur = ++done;
            long now = System.nanoTime();
            if (cur==total || now - lastPrint > 200_000_000L) { // 200ms 刷新
                print(cur);
                lastPrint = now;
            }
        }
        void print(int cur){
            double pct = total>0 ? (cur*1.0/total) : 1.0;
            int fill = (int)Math.round(pct * barLen);
            StringBuilder bar = new StringBuilder();
            for (int i=0;i<barLen;i++) bar.append(i<fill ? '█' : ' ');
            double sec = (System.nanoTime() - start)/1e9;
            double spd = sec>0 ? cur/sec : 0.0; // samples/sec
            double remain = spd>0 ? (total - cur)/spd : 0.0;
            System.out.printf("\r[%s] %d/%d  %5.1f%%  %.1f/s  ETA %s",
                    bar, cur, total, pct*100.0, spd, formatSec(remain));
        }
        void done(){ print(total); System.out.println(); }
        void finishWithError(){ System.out.println("\n❌ 生成过程中出现错误。"); }
        static String formatSec(double s){
            long sec = Math.max(0, (long)Math.round(s));
            long m = sec/60, ss = sec%60;
            return String.format("%02d:%02d", m, ss);
        }
    }

    // -------------------- 打印/统计 --------------------
    static <T> void inc(Map<T,Integer> m, T k){ m.put(k, m.getOrDefault(k,0)+1); }
    static int sum(Map<?,Integer> m){ int s=0; for (int v : m.values()) s+=v; return s; }

    static void printTable(String title, Map<String,Integer> m, int total){
        System.out.println("== " + title + " ==");
        int w = 0; for (String k : m.keySet()) w = Math.max(w, k.length());
        for (Map.Entry<String,Integer> e : m.entrySet()){
            double pct = total>0 ? e.getValue()*100.0/total : 0.0;
            System.out.printf("  %-" + w + "s : %6d  (%5.1f%%)\n", e.getKey(), e.getValue(), pct);
        }
        System.out.println();
    }
    static void printTableInt(String title, Map<Integer,Integer> m, int total){
        System.out.println("== " + title + " ==");
        List<Integer> keys = new ArrayList<>(m.keySet());
        Collections.sort(keys);
        for (Integer k : keys){
            int v = m.get(k);
            double pct = total>0 ? v*100.0/total : 0.0;
            System.out.printf("  %2d : %6d  (%5.1f%%)\n", k, v, pct);
        }
        System.out.println();
    }

    // -------------------- 统计落盘 --------------------
    static void writeStatsJson(
            Path outDir, String[] argv,
            Map<String,Integer> sizePlan, Map<String,Integer> taskPlan,
            Map<String,Integer> textMix,  Map<String,Integer> lenPlan,
            Map<String,Integer> arithLen,
            Map<String,Integer> sizeCnt, Map<String,Integer> taskCnt,
            Map<String,Integer> typeCnt, Map<Integer,Integer> tlenCnt,
            Map<Integer,Integer> alenCnt, Map<String,Integer> opCntFinal,
            int threads, int total) throws IOException {

        Path json = outDir.resolve("stats.json");
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        // 参数
        sb.append("  \"args\": ").append(jsonEscape(String.join(" ", argv))).append(",\n");
        sb.append("  \"threads\": ").append(threads).append(",\n");
        sb.append("  \"total\": ").append(total).append(",\n");
        // 计划
        sb.append("  \"plan\": {\n");
        sb.append("    \"sizes\": ").append(mapToJson(sizePlan)).append(",\n");
        sb.append("    \"task\": ").append(mapToJson(taskPlan)).append(",\n");
        sb.append("    \"text_mix\": ").append(mapToJson(textMix)).append(",\n");
        sb.append("    \"len\": ").append(mapToJson(lenPlan)).append(",\n");
        sb.append("    \"arith_len\": ").append(mapToJson(arithLen)).append("\n");
        sb.append("  },\n");
        // 实际
        sb.append("  \"actual\": {\n");
        sb.append("    \"sizes\": ").append(mapToJson(sizeCnt)).append(",\n");
        sb.append("    \"task\": ").append(mapToJson(taskCnt)).append(",\n");
        sb.append("    \"text_types\": ").append(mapToJson(typeCnt)).append(",\n");
        sb.append("    \"text_len\": ").append(mapIntToJson(tlenCnt)).append(",\n");
        sb.append("    \"arith_len\": ").append(mapIntToJson(alenCnt)).append(",\n");
        sb.append("    \"arith_ops_actual\": ").append(mapToJson(opCntFinal)).append("\n");
        sb.append("  }\n");
        sb.append("}\n");
        Files.write(json, sb.toString().getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    }

    static void writeStatsTxt(
            Path outDir,
            Map<String,Integer> sizeCnt, Map<String,Integer> taskCnt,
            Map<String,Integer> typeCnt, Map<Integer,Integer> tlenCnt,
            Map<Integer,Integer> alenCnt, Map<String,Integer> opCntFinal,
            int total) throws IOException {

        Path txt = outDir.resolve("stats.txt");
        StringBuilder sb = new StringBuilder();
        sb.append("== 尺寸分布 ==\n").append(tableToString(sizeCnt, total));
        sb.append("\n== 任务分布 ==\n").append(tableToString(taskCnt, total));
        sb.append("\n== 文本类型分布（text-mix） ==\n").append(tableToString(typeCnt, sum(typeCnt)));
        sb.append("\n== 文本长度分布（len） ==\n").append(tableIntToString(tlenCnt, sum(tlenCnt)));
        sb.append("\n== 算术参与数分布（arith-len） ==\n").append(tableIntToString(alenCnt, sum(alenCnt)));
        sb.append("\n== 算术操作符分布（实际） ==\n").append(tableToString(opCntFinal, sum(opCntFinal)));
        Files.write(txt, sb.toString().getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    }

    static String tableToString(Map<String,Integer> m, int total){
        int w = 0; for (String k : m.keySet()) w = Math.max(w, k.length());
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String,Integer> e : m.entrySet()){
            double pct = total>0 ? e.getValue()*100.0/total : 0.0;
            sb.append(String.format("  %-" + w + "s : %6d  (%5.1f%%)\n", e.getKey(), e.getValue(), pct));
        }
        return sb.toString();
    }
    static String tableIntToString(Map<Integer,Integer> m, int total){
        List<Integer> keys = new ArrayList<>(m.keySet());
        Collections.sort(keys);
        StringBuilder sb = new StringBuilder();
        for (Integer k : keys){
            int v = m.get(k);
            double pct = total>0 ? v*100.0/total : 0.0;
            sb.append(String.format("  %2d : %6d  (%5.1f%%)\n", k, v, pct));
        }
        return sb.toString();
    }

    static String jsonEscape(String s){
        StringBuilder sb = new StringBuilder("\"");
        for (char c : s.toCharArray()){
            switch (c){
                case '\\': sb.append("\\\\"); break;
                case '\"': sb.append("\\\""); break;
                case '\n': sb.append("\\n"); break;
                case '\r': sb.append("\\r"); break;
                case '\t': sb.append("\\t"); break;
                default:
                    if (c < 32) sb.append(String.format("\\u%04x", (int)c));
                    else sb.append(c);
            }
        }
        sb.append("\"");
        return sb.toString();
    }
    static String mapToJson(Map<String,? extends Number> m){
        StringBuilder sb = new StringBuilder("{");
        boolean first=true;
        for (Map.Entry<String,? extends Number> e : m.entrySet()){
            if (!first) sb.append(","); first=false;
            sb.append(jsonEscape(e.getKey())).append(": ").append(e.getValue());
        }
        sb.append("}");
        return sb.toString();
    }
    static String mapIntToJson(Map<Integer,? extends Number> m){
        StringBuilder sb = new StringBuilder("{");
        boolean first=true;
        for (Map.Entry<Integer,? extends Number> e : m.entrySet()){
            if (!first) sb.append(","); first=false;
            sb.append("\"").append(e.getKey()).append("\": ").append(e.getValue());
        }
        sb.append("}");
        return sb.toString();
    }

    // -------------------- 分布构造（严格配额+洗牌） --------------------
    static <T> List<T> buildListFromPercent(int total, LinkedHashMap<T,Integer> plan){
        if (plan.isEmpty()) throw new IllegalArgumentException("分布为空");
        LinkedHashMap<T,Integer> cnt = new LinkedHashMap<>();
        int acc = 0, i=0;
        for (Map.Entry<T,Integer> e : plan.entrySet()){
            int last = (i++ == plan.size()-1) ? 1 : 0;
            int c = last==1 ? (total-acc) : allocFromPercent(total, e.getValue());
            acc += c; cnt.put(e.getKey(), c);
        }
        ArrayList<T> list = new ArrayList<>(total);
        for (Map.Entry<T,Integer> e : cnt.entrySet())
            for (int k=0;k<e.getValue();k++) list.add(e.getKey());
        Collections.shuffle(list, new Random());
        return list;
    }
    static List<Integer> buildLenList(int total, LinkedHashMap<String,Integer> plan){
        LinkedHashMap<Integer,Integer> p = new LinkedHashMap<>();
        for (Map.Entry<String,Integer> e : plan.entrySet()) p.put(parseInt(e.getKey(), 4), e.getValue());
        if (p.isEmpty()) { p.put(4,50); p.put(5,25); p.put(6,25); }
        LinkedHashMap<Integer,Integer> cnt = new LinkedHashMap<>();
        int acc=0, i=0;
        for (Map.Entry<Integer,Integer> e : p.entrySet()){
            int last = (i++ == p.size()-1) ? 1 : 0;
            int c = last==1 ? (total-acc) : allocFromPercent(total, e.getValue());
            acc += c; cnt.put(e.getKey(), c);
        }
        ArrayList<Integer> list = new ArrayList<>(total);
        for (Map.Entry<Integer,Integer> e : cnt.entrySet())
            for (int k=0;k<e.getValue();k++) list.add(e.getKey());
        Collections.shuffle(list, new Random());
        return list;
    }
    static int countOf(List<String> list, String v){ int c=0; for (String s: list) if (v.equals(s)) c++; return c; }

    // -------------------- 类型/操作符 --------------------
    static int mapCharType(String tp){
        // easy-captcha: 1混合 2纯数字 3纯字母 4大写 5小写 6数字+大写 7数字+小写
        switch (tp){
            case "spec": return 1;
            case "spec_num": return 2;
            case "spec_char": return 3;
            case "spec_upper": return 4;
            case "spec_lower": return 5;
            case "spec_numupper": return 6;
            case "spec_numlower": return 7;
            default: return 1;
        }
    }

    static String extractOp(String expr){
        if (expr==null) return null;
        if (expr.contains("+")) return "+";
        if (expr.contains("-")) return "-";
        if (expr.contains("×")) return "×";
        if (expr.contains("÷")) return "÷";
        if (expr.contains("*")) return "*";
        if (expr.contains("/")) return "/";
        return null;
    }

    // -------------------- 字体计划（含算术兜底） --------------------
    static class FontPlan {
        enum Mode { NONE, BUILTIN, LIST }
        Mode mode = Mode.NONE;
        List<String> nameList = new ArrayList<>();
        List<Font> builtin = new ArrayList<>();
        // 预解析池（加速）
        List<Font> arithPool = new ArrayList<>(); // 覆盖 0123456789+-×÷=？
    }

    static FontPlan parseFontPlan(String[] argv){
        FontPlan fp = new FontPlan();
        for (int i=0;i<argv.length-1;i++){
            if ("--fonts".equalsIgnoreCase(argv[i])){
                String v = argv[i+1];
                if ("builtin".equalsIgnoreCase(v)){
                    fp.mode = FontPlan.Mode.BUILTIN;
                    fp.builtin = loadBuiltinFonts();
                } else {
                    fp.mode = FontPlan.Mode.LIST;
                    fp.nameList = new ArrayList<>(Arrays.asList(v.split(",")));
                }
            }
        }
        return fp;
    }

    static final String[] BUILTIN_TTFS = new String[]{
        "actionj.ttf","epilog.ttf","fresnel.ttf","headache.ttf",
        "lexo.ttf","prefix.ttf","progbot.ttf","ransom.ttf","robot.ttf","scandal.ttf"
    };
    static List<Font> loadBuiltinFonts(){
        List<Font> list = new ArrayList<>();
        ClassLoader cl = EasyCaptchaSmartGen.class.getClassLoader();
        for (String name : BUILTIN_TTFS){
            try (InputStream is = cl.getResourceAsStream(name)) {
                if (is != null){
                    Font f = Font.createFont(Font.TRUETYPE_FONT, is);
                    list.add(f);
                }
            } catch (Exception ignore) {}
        }
        return list;
    }

    static final String ARITH_TEST = "0123456789+-×÷=？";
    static final String[] SAFE_ARITH_FONTS = new String[]{
        "Microsoft YaHei","SimSun","SimHei","Segoe UI Symbol",
        "Arial Unicode MS","Arial","Tahoma",
        "DejaVu Sans","Liberation Sans","Noto Sans","Helvetica"
    };
    static boolean fontSupports(Font f, String sample){
        try { return f != null && f.canDisplayUpTo(sample) == -1; }
        catch (Throwable ignore){ return false; }
    }

    static void prepareFontPools(FontPlan fp){
        // 构建算术可用字体池（一次性），后续仅 deriveFont 改大小
        HashSet<String> seen = new HashSet<>();
        // 1) 用户自定义优先
        if (fp.mode == FontPlan.Mode.LIST){
            for (String name : fp.nameList){
                String nm = name.trim();
                if (nm.isEmpty()) continue;
                if (seen.add(nm)){
                    Font f = new Font(nm, Font.PLAIN, 32);
                    if (fontSupports(f, ARITH_TEST)) fp.arithPool.add(f);
                }
            }
        }
        // 2) 内置字体（大多不支持算术符号，能显示就收）
        if (fp.mode == FontPlan.Mode.BUILTIN){
            for (Font base : fp.builtin){
                if (base!=null){
                    Font f = base.deriveFont(Font.PLAIN, 32f);
                    if (fontSupports(f, ARITH_TEST)) fp.arithPool.add(f);
                }
            }
        }
        // 3) 安全白名单兜底
        for (String name : SAFE_ARITH_FONTS){
            if (seen.add(name)){
                Font f = new Font(name, Font.PLAIN, 32);
                if (fontSupports(f, ARITH_TEST)) fp.arithPool.add(f);
            }
        }
        // 若仍为空，则后续走 JRE 默认回退（不设置）
    }

    static void applyFontIfAny(Captcha c, FontPlan fp, int H, int idx){
        try{
            float sz = Math.max(18f, H * 0.72f);
            if (fp.mode == FontPlan.Mode.BUILTIN && !fp.builtin.isEmpty()){
                Font base = fp.builtin.get(idx % fp.builtin.size());
                c.setFont(base.deriveFont(Font.PLAIN, sz));
            }else if (fp.mode == FontPlan.Mode.LIST && !fp.nameList.isEmpty()){
                String name = fp.nameList.get(idx % fp.nameList.size()).trim();
                c.setFont(new Font(name, Font.BOLD, (int)sz));
            }
        }catch(Throwable ignore){}
    }

    static void applyFontForArith(Captcha c, FontPlan fp, int imgH, int idx){
        try{
            float sz = Math.max(18f, imgH * 0.72f);
            if (!fp.arithPool.isEmpty()){
                Font base = fp.arithPool.get(idx % fp.arithPool.size());
                c.setFont(base.deriveFont(Font.BOLD, sz));
                return;
            }
            // 没有可用池则尝试常规策略
            applyFontIfAny(c, fp, imgH, idx);
        }catch(Throwable ignore){}
    }

    // -------------------- 依赖自检（算术需） --------------------
    static void ensureArithDepsIfNeeded(int arithCount){
        if (arithCount <= 0) return;
        try { Class.forName("org.openjdk.nashorn.api.scripting.NashornScriptEngineFactory"); }
        catch (ClassNotFoundException e) {
            throw new RuntimeException("缺少 nashorn-core（JDK15+ 必需）。请加入 nashorn-core-15.4.jar。", e);
        }
        String[][] need = {
            {"org.objectweb.asm.Type",                          "asm-9.6.jar"},
            {"org.objectweb.asm.commons.Method",                "asm-commons-9.6.jar"},
            {"org.objectweb.asm.util.Printer",                  "asm-util-9.6.jar"},
            {"org.objectweb.asm.tree.ClassNode",                "asm-tree-9.6.jar"},
            {"org.objectweb.asm.tree.analysis.Analyzer",        "asm-analysis-9.6.jar"}
        };
        for (String[] n : need){
            try { Class.forName(n[0]); }
            catch (ClassNotFoundException e) {
                throw new RuntimeException("缺少 ASM 依赖：" + n[0] + "（请加入 " + n[1] + "）", e);
            }
        }
    }

    // -------------------- 工具 --------------------
    static String getArg(String[] args, String key, String def){
        for (int i=0;i<args.length-1;i++) if (key.equalsIgnoreCase(args[i])) return args[i+1];
        return def;
    }
    static LinkedHashMap<String,Integer> parseKVPlan(String s){
        LinkedHashMap<String,Integer> m = new LinkedHashMap<>();
        if (s==null || s.trim().isEmpty()) return m;
        for (String part : s.split(",")){
            String[] kv = part.trim().split(":");
            if (kv.length==2) m.put(kv[0].trim(), parseInt(kv[1].trim(), 0));
        }
        return m;
    }
    static LinkedHashMap<String,Integer> parseSizesPlan(String s){ return parseKVPlan(s); }
    static int allocFromPercent(int total, int percent){
        return Math.max(0, Math.min(total, (int)Math.round(total * (percent/100.0))));
    }
    static int parseInt(String s, int d){ try { return Integer.parseInt(s); } catch(Exception e){ return d; } }
    static String uuid(){ return java.util.UUID.randomUUID().toString().replace("-",""); }
}
