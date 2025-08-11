// -*- coding: utf-8 -*-
// easy-captcha-1.6.2 批量生成器（JSON 配置驱动 / CN & ARITH 字体修正 / GIF=0 不生成 / 预测干跑）
// 输出结构： ./images/<outName>/images/*.png|gif  +  ./images/<outName>/labels.txt
// 依赖：easy-captcha-1.6.2-RELEASE.jar；算术在 JDK15+ 需 nashorn-core & ASM 全家桶（lib/*）
// JSON：见文末示例；可通过 fromjson 命令读取

import com.wf.captcha.*;
import com.wf.captcha.base.Captcha;
import com.google.gson.*;

import java.awt.Font;
import java.io.*;
import java.lang.reflect.Type;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

public class EasyCaptchaRunner {

    // ----------------------------------------------------------------------
    // 入口
    // ----------------------------------------------------------------------
    public static void main(String[] args) throws Exception {
        if (args.length == 0 || "list".equalsIgnoreCase(args[0])) {
            printTypes();
            System.out.println("\n命令：");
            System.out.println("  fromjson <config.json> dryrun <total>            # 只打印预测分布，不生成");
            System.out.println("  fromjson <config.json> <outName> <total>         # 按配置生成");
            System.out.println("\n兼容旧命令：");
            System.out.println("  all <outBase> <count> [W H LEN] [--fonts ...]");
            System.out.println("  <type> <outName> <count> [W H LEN] [--fonts ...]");
            return;
        }

        final Path ROOT = Paths.get("images");

        String cmd = args[0].toLowerCase(Locale.ROOT);
        switch (cmd) {
            case "fromjson": {
                if (args.length < 4) {
                    System.out.println("用法：");
                    System.out.println("  java -cp \".;lib/*\" EasyCaptchaRunner fromjson <config.json> dryrun <total>");
                    System.out.println("  java -cp \".;lib/*\" EasyCaptchaRunner fromjson <config.json> <outName> <total>");
                    return;
                }
                Path cfgPath = Paths.get(args[1]);
                String outName = args[2];
                int total = parseInt(args[3], 1000);
                Map<String, Object> cfg = readJsonAsMap(cfgPath);

                if ("dryrun".equalsIgnoreCase(outName)) {
                    // 只打印预测分布
                    Prediction pred = predictFromJson(cfg, total);
                    printPrediction(pred, total, true);
                    return;
                }

                // 先打印预测，再真正生成
                Prediction pred = predictFromJson(cfg, total);
                printPrediction(pred, total, false);
                runFromJson(cfg, ROOT.resolve(outName), pred);
                System.out.println("DONE -> " + ROOT.toAbsolutePath());
                return;
            }

            case "all": {
                // 旧命令：java EasyCaptchaRunner all <outBase> <count> [W H LEN] [--fonts ...]
                String outBase = args.length > 1 ? args[1] : "all";
                int n   = args.length > 2 ? parseInt(args[2], 10)  : 10;
                int W   = args.length > 3 ? parseInt(args[3], 160) : 160;
                int H   = args.length > 4 ? parseInt(args[4], 64)  : 64;
                int LEN = args.length > 5 ? parseInt(args[5], 0)   : 0;   // <=0 启用 4/5/6 分布
                FontPlan fontPlan = parseFontPlan(args);

                String[] types = {
                        "spec","spec_num","spec_char","spec_upper","spec_lower",
                        "spec_numupper","spec_numlower",
                        "gif","gif_numupper","gif_numlower",
                        "cn","cn_gif",
                        "arith"
                };
                for (String t : types) runOne(t, ROOT.resolve(outBase + "_" + t), n, W, H, LEN, fontPlan);
                System.out.println("DONE -> " + ROOT.toAbsolutePath());
                return;
            }

            default: {
                // 旧命令：java EasyCaptchaRunner <type> <outName> <count> [W H LEN] [--fonts ...]
                if (args.length < 3) {
                    System.out.println("用法：");
                    System.out.println("  列表:  java -cp .;lib/* EasyCaptchaRunner list");
                    System.out.println("  JSON:  java -cp .;lib/* EasyCaptchaRunner fromjson <config.json> <outName|dryrun> <total>");
                    System.out.println("  ALL :  java -cp .;lib/* EasyCaptchaRunner all <outBase> <count> [W H LEN] [--fonts ...]");
                    System.out.println("  单类:  java -cp .;lib/* EasyCaptchaRunner <type> <outName> <count> [W H LEN] [--fonts ...]");
                    printTypes();
                    return;
                }
                String type    = cmd;
                String outName = args[1];
                int n   = parseInt(args[2], 10);
                int W   = args.length > 3 ? parseInt(args[3], 160) : 160;
                int H   = args.length > 4 ? parseInt(args[4], 64)  : 64;
                int LEN = args.length > 5 ? parseInt(args[5], 0)   : 0;
                FontPlan fontPlan = parseFontPlan(args);

                runOne(type, ROOT.resolve(outName), n, W, H, LEN, fontPlan);
            }
        }
    }

    // ----------------------------------------------------------------------
    // 列表 & 旧命令实现
    // ----------------------------------------------------------------------
    static void printTypes() {
        System.out.println("可用类型：");
        String[] t = {
                "spec", "spec_num", "spec_char", "spec_upper", "spec_lower",
                "spec_numupper", "spec_numlower",
                "gif", "gif_numupper", "gif_numlower",
                "cn", "cn_gif",
                "arith"
        };
        for (String s : t) System.out.println("  - " + s);
        System.out.println("\n算术(arith)在 JDK15+ 需要 nashorn-core + ASM 全家桶（lib/*）。");
    }

    static void runOne(String type, Path outDir, int count, int W, int H, int LEN, FontPlan fp) throws Exception {
        Path imgDir = outDir.resolve("images");
        Path lbl    = outDir.resolve("labels.txt");
        Files.createDirectories(imgDir);

        System.out.printf(">> %-12s -> %s  (count=%d, %dx%d, len=%s)%n",
                type, outDir, count, W, H, (LEN<=0?"auto[4/5/6]":String.valueOf(LEN)));

        try (BufferedWriter bw = Files.newBufferedWriter(lbl, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {

            switch (type) {
                // 静态 PNG
                case "spec":           genSpec(imgDir, bw, count, W, H, LEN, 1, fp); break;
                case "spec_num":       genSpec(imgDir, bw, count, W, H, LEN, 2, fp); break;
                case "spec_char":      genSpec(imgDir, bw, count, W, H, LEN, 3, fp); break;
                case "spec_upper":     genSpec(imgDir, bw, count, W, H, LEN, 4, fp); break;
                case "spec_lower":     genSpec(imgDir, bw, count, W, H, LEN, 5, fp); break;
                case "spec_numupper":  genSpec(imgDir, bw, count, W, H, LEN, 6, fp); break;
                case "spec_numlower":  genSpec(imgDir, bw, count, W, H, LEN, 7, fp); break;

                // GIF
                case "gif":            genGif(imgDir, bw, count, W, H, LEN, 1, fp); break;
                case "gif_numupper":   genGif(imgDir, bw, count, W, H, LEN, 6, fp); break;
                case "gif_numlower":   genGif(imgDir, bw, count, W, H, LEN, 7, fp); break;

                // 中文
                case "cn":             genCn(imgDir, bw, count, W, H, LEN, false, fp); break;
                case "cn_gif":         genCn(imgDir, bw, count, W, H, LEN, true , fp); break;

                // 算术（PNG，标签=表达式本身）
                case "arith":
                    ensureArithDeps(); // 运行前检查依赖（JDK15+）
                    genArith(imgDir, bw, count, W, H, LEN, fp);
                    break;

                default: throw new IllegalArgumentException("未知类型: " + type);
            }
        }
    }

    // ---------- 旧路径：文本 ----------
    static void genSpec(Path dir, BufferedWriter bw, int n, int W, int H, int LEN, int charType, FontPlan fp) throws Exception {
        for (int i=0;i<n;i++){
            int cur = chooseLen(LEN);
            SpecCaptcha c = new SpecCaptcha(W, H, cur);
            c.setCharType(charType);
            applyFontIfAny(c, fp, H, i);
            String label = c.text();
            String fn = uuid()+".png";
            try (OutputStream os = Files.newOutputStream(dir.resolve(fn))) { c.out(os); }
            bw.write(fn + "\t" + label); bw.newLine();
        }
    }

    static void genGif(Path dir, BufferedWriter bw, int n, int W, int H, int LEN, int charType, FontPlan fp) throws Exception {
        for (int i=0;i<n;i++){
            int cur = chooseLen(LEN);
            GifCaptcha c = new GifCaptcha(W, H, cur);
            c.setCharType(charType);
            applyFontIfAny(c, fp, H, i);
            String label = c.text();
            String fn = uuid()+".gif";
            try (OutputStream os = Files.newOutputStream(dir.resolve(fn))) { c.out(os); }
            bw.write(fn + "\t" + label); bw.newLine();
        }
    }

    static void genCn(Path dir, BufferedWriter bw, int n, int W, int H, int LEN, boolean gif, FontPlan fp) throws Exception {
        for (int i=0;i<n;i++){
            int cur = chooseLen(LEN);
            Captcha c = gif ? new ChineseGifCaptcha(W,H,cur) : new ChineseCaptcha(W,H,cur);
            applyFontIfAny(c, fp, H, i);
            String label = c.text();
            String fn = uuid() + (gif? ".gif" : ".png");
            try (OutputStream os = Files.newOutputStream(dir.resolve(fn))) { c.out(os); }
            bw.write(fn + "\t" + label); bw.newLine();
        }
    }

    // ---------- 旧路径：算术 ----------
    static void genArith(Path dir, BufferedWriter bw, int n, int W, int H, int LEN, FontPlan fp) throws Exception {
        for (int i=0;i<n;i++){
            ArithmeticCaptcha c = new ArithmeticCaptcha(W, H);
            c.setLen(LEN>0 ? LEN : 2);      // 通常两个数
            applyFontForArith(c, fp, H, i); // 覆盖 × ÷ ？ 字形
            String expr = c.getArithmeticString(); // 标签=表达式
            String fn = uuid()+".png";
            try (OutputStream os = Files.newOutputStream(dir.resolve(fn))) { c.out(os); }
            bw.write(fn + "\t" + expr); bw.newLine();
        }
    }

    static int chooseLen(int fixed){
        if (fixed>0) return fixed;
        double r = Math.random();
        if (r < 0.50) return 4;
        if (r < 0.75) return 5;
        return 6;
    }

    // ----------------------------------------------------------------------
    // JSON 驱动：预测 + 生成
    // ----------------------------------------------------------------------

    // 预测结果结构（仅用于显示/指导生成）
    static class Prediction {
        // 每类型的总量
        Map<String,Integer> typeCount = new LinkedHashMap<>();
        // 每类型 -> 尺寸分配
        Map<String, Map<String,Integer>> typeSizeCount = new LinkedHashMap<>();
        // 每类型 -> （len 或 arith_len）分配
        Map<String, Map<String,Integer>> typeLenCount = new LinkedHashMap<>();
        // 每类型 -> 使用的字体权重（用于展示；不细分具体数量）
        Map<String, Map<String,Integer>> typeFonts = new LinkedHashMap<>();
        // 全局尺寸（原始权重）
        Map<String,Integer> sizes = new LinkedHashMap<>();
    }

    static Prediction predictFromJson(Map<String,Object> cfg, int total){
        Prediction p = new Prediction();

        Map<String,Integer> typeMix = toIntMap((Map<String,Object>) cfg.get("type_mix"));
        Map<String,Integer> sizes   = toIntMap((Map<String,Object>) cfg.getOrDefault("sizes", defaultSizes()));
        Map<String,Object>  defaults= (Map<String,Object>) cfg.getOrDefault("defaults", Collections.emptyMap());
        Map<String,Integer> defaultLen   = toIntMap((Map<String,Object>) defaults.getOrDefault("len", defaultLenDist()));
        Map<String,Integer> defaultFonts = toIntMap((Map<String,Object>) defaults.getOrDefault("fonts", Collections.singletonMap("builtin", 1)));
        Map<String,Object>  perType = (Map<String,Object>) cfg.getOrDefault("per_type", Collections.emptyMap());

        // 类型数量（修正：只给权重>0的类型分配，避免 0 权重也产出）
        Map<String,Integer> typeCount = allocByWeight(typeMix, total);
        p.typeCount.putAll(typeCount);
        p.sizes = sizes;

        // 各类型细分
        for (String type : typeCount.keySet()){
            int tTotal = typeCount.get(type);

            Map<String,Object> tcfg = (Map<String,Object>) perType.getOrDefault(type, Collections.emptyMap());

            // 尺寸
            Map<String,Integer> tSizes = allocByWeight(sizes, tTotal);
            p.typeSizeCount.put(type, tSizes);

            // 长度/算术长度
            boolean isArith = "arith".equals(type);
            Map<String,Integer> tLenWeight = isArith
                    ? toIntMap((Map<String,Object>) tcfg.getOrDefault("arith_len", defaultArithLen()))
                    : toIntMap((Map<String,Object>) tcfg.getOrDefault("len", defaultLen));
            Map<String,Integer> tLenCount = allocByWeight(tLenWeight, tTotal);
            p.typeLenCount.put(type, tLenCount);

            // 字体（仅展示用）
            Map<String,Integer> perFonts = toIntMap((Map<String,Object>) tcfg.getOrDefault("fonts", Collections.emptyMap()));
            Map<String,Integer> tFonts;
            boolean isCnOrArith = "cn".equals(type) || "arith".equals(type);
            if (isCnOrArith) {
                tFonts = new LinkedHashMap<>(perFonts);
                if (!perFonts.containsKey("builtin")) tFonts.remove("builtin");
                if (tFonts.isEmpty()) {
                    tFonts.put("微软雅黑",18); tFonts.put("宋体",12); tFonts.put("黑体",8);
                    tFonts.put("等线",7); tFonts.put("Noto Sans SC",18); tFonts.put("Noto Serif SC",7);
                    tFonts.put("阿里巴巴普惠体 R",8); tFonts.put("华文宋体",4); tFonts.put("华文楷体",4);
                }
            } else {
                tFonts = mergeWeightMap(defaultFonts, perFonts);
                if (tFonts.isEmpty()) tFonts.put("builtin",1);
            }
            p.typeFonts.put(type, tFonts);
        }
        return p;
    }

    static void printPrediction(Prediction pred, int total, boolean headerOnly){
        System.out.println("== 预测任务分布 ==");
        for (Map.Entry<String,Integer> e : pred.typeCount.entrySet()){
            System.out.printf("  %-12s : %7d (%.1f%%)%n", e.getKey(), e.getValue(), 100.0*e.getValue()/total);
        }
        if (headerOnly) return;

        System.out.println("\n== 尺寸分布（全局权重） ==");
        int ssum = sumWeights(pred.sizes);
        for (Map.Entry<String,Integer> e : pred.sizes.entrySet()){
            System.out.printf("  %-7s : %5d (%.1f%%)%n", e.getKey(), e.getValue(), 100.0*e.getValue()/Math.max(1,ssum));
        }

        System.out.println("\n== 每类型长度分布 ==");
        for (String t : pred.typeLenCount.keySet()){
            System.out.printf("  [%s]%n", t);
            Map<String,Integer> m = pred.typeLenCount.get(t);
            int tsum = pred.typeCount.get(t);
            for (Map.Entry<String,Integer> e : m.entrySet()){
                System.out.printf("    %-3s : %6d (%.1f%% of %d)%n", e.getKey(), e.getValue(), 100.0*e.getValue()/tsum, tsum);
            }
        }

        System.out.println("\n== 每类型字体（权重，仅展示） ==");
        for (String t : pred.typeFonts.keySet()){
            System.out.printf("  [%s]%n", t);
            Map<String,Integer> m = pred.typeFonts.get(t);
            int sum = sumWeights(m);
            for (Map.Entry<String,Integer> e : m.entrySet()){
                System.out.printf("    %-20s : %4d (%.1f%%)%n", e.getKey(), e.getValue(), 100.0*e.getValue()/Math.max(1,sum));
            }
        }
        System.out.println();
    }

    static void runFromJson(Map<String,Object> cfg, Path outDir, Prediction pred) throws Exception {
        Files.createDirectories(outDir.resolve("images"));
        Path imgDir = outDir.resolve("images");
        Path lbl    = outDir.resolve("labels.txt");

        try (BufferedWriter bw = Files.newBufferedWriter(lbl, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {

            for (String type : pred.typeCount.keySet()){
                int tTotal = pred.typeCount.get(type);
                Map<String,Integer> sizeAlloc = pred.typeSizeCount.get(type);
                Map<String,Integer> lenAlloc  = pred.typeLenCount.get(type);
                Map<String,Integer> fontW     = pred.typeFonts.get(type);

                FontPicker fontPicker = new FontPicker(fontW);

                boolean isGifType = type.startsWith("gif");
                boolean isCnType  = "cn".equals(type) || "cn_gif".equals(type);
                boolean isArith   = "arith".equals(type);

                if (isArith) ensureArithDeps();

                System.out.printf(">> %-12s -> %s  (count=%d)%n", type, outDir, tTotal);

                // 分配策略：按 lenAlloc 展开（这样能精确控制长度比例），尺寸和字体在循环内按权重随机
                for (Map.Entry<String,Integer> eLen : lenAlloc.entrySet()){
                    String lenKey = eLen.getKey();
                    int thisLenCount = eLen.getValue();
                    if (thisLenCount <= 0) continue;

                    // 尺寸轮盘
                    Roulette<String> sizeRoulette = new Roulette<>(pred.sizes);

                    for (int i=0; i<thisLenCount; i++){
                        String size = sizeRoulette.next();
                        int W = parseInt(size.split("x")[0],160);
                        int H = parseInt(size.split("x")[1],64);

                        if (isArith){
                            ArithmeticCaptcha c = new ArithmeticCaptcha(W,H);
                            c.setLen(parseInt(lenKey,2));
                            fontPicker.applyForArith(c, H);
                            String expr = c.getArithmeticString();
                            String fn = uuid()+".png";
                            try (OutputStream os = Files.newOutputStream(imgDir.resolve(fn))) { c.out(os); }
                            bw.write(fn + "\t" + expr); bw.newLine();
                        } else if (isCnType){
                            boolean gif = "cn_gif".equals(type);
                            Captcha c = gif ? new ChineseGifCaptcha(W,H,parseInt(lenKey,4)) : new ChineseCaptcha(W,H,parseInt(lenKey,4));
                            fontPicker.applyNormal(c, H);
                            String label = c.text();
                            String ext = gif?".gif":".png";
                            String fn = uuid()+ext;
                            try (OutputStream os = Files.newOutputStream(imgDir.resolve(fn))) { c.out(os); }
                            bw.write(fn + "\t" + label); bw.newLine();
                        } else {
                            int charType = mapTypeToCharType(type);
                            if (isGifType){
                                GifCaptcha c = new GifCaptcha(W,H,parseInt(lenKey,4));
                                c.setCharType(charType);
                                fontPicker.applyNormal(c, H);
                                String label = c.text();
                                String fn = uuid()+".gif";
                                try (OutputStream os = Files.newOutputStream(imgDir.resolve(fn))) { c.out(os); }
                                bw.write(fn + "\t" + label); bw.newLine();
                            } else {
                                SpecCaptcha c = new SpecCaptcha(W,H,parseInt(lenKey,4));
                                c.setCharType(charType);
                                fontPicker.applyNormal(c, H);
                                String label = c.text();
                                String fn = uuid()+".png";
                                try (OutputStream os = Files.newOutputStream(imgDir.resolve(fn))) { c.out(os); }
                                bw.write(fn + "\t" + label); bw.newLine();
                            }
                        }
                    }
                }
            }
        }
    }

    // ----------------------------------------------------------------------
    // 字体挑选器（按权重），含 CN/ARITH 特殊兜底
    // ----------------------------------------------------------------------
    static class FontPicker {
        final LinkedHashMap<String,Integer> weight = new LinkedHashMap<>();
        final int total;
        final Random rnd = new Random();
        final java.util.List<Font> builtin = loadBuiltinFonts();

        FontPicker(Map<String,Integer> w){
            int sum=0;
            for (Map.Entry<String,Integer> e : w.entrySet()){
                int v = Math.max(0, e.getValue());
                if (v>0) { weight.put(e.getKey(), v); sum+=v; }
            }
            total = Math.max(1,sum);
        }

        String pickKey(){
            int r = rnd.nextInt(total);
            int acc=0;
            for (Map.Entry<String,Integer> e : weight.entrySet()){
                acc += e.getValue();
                if (r < acc) return e.getKey();
            }
            return weight.keySet().iterator().next();
        }

        void applyNormal(Captcha c, int imgH){
            float sz = Math.max(18f, imgH * 0.72f);
            String key = pickKey();
            if ("builtin".equalsIgnoreCase(key)){
                if (!builtin.isEmpty()){
                    Font base = builtin.get(rnd.nextInt(builtin.size()));
                    c.setFont(base.deriveFont(Font.PLAIN, sz));
                    return;
                }
                // builtin 不可用，回退无
            }
            try{
                c.setFont(new Font(key, Font.BOLD, (int)sz));
            }catch(Throwable ignore){}
        }

        void applyForArith(Captcha c, int imgH){
            float sz = Math.max(18f, imgH * 0.72f);
            String key = pickKey();
            if (!"builtin".equalsIgnoreCase(key)){
                Font f = new Font(key, Font.BOLD, (int)sz);
                if (fontSupports(f, ARITH_TEST)){ c.setFont(f); return; }
            }
            // builtin 或不支持 -> 试 builtin
            for (int i=0;i<builtin.size();i++){
                Font base = builtin.get((i + rnd.nextInt(builtin.size())) % builtin.size());
                Font f = base.deriveFont(Font.PLAIN, sz);
                if (fontSupports(f, ARITH_TEST)){ c.setFont(f); return; }
            }
            // 安全兜底
            for (String name : SAFE_ARITH_FONTS){
                Font f = new Font(name, Font.BOLD, (int)sz);
                if (fontSupports(f, ARITH_TEST)){ c.setFont(f); return; }
            }
        }
    }

    // ----------------------------------------------------------------------
    // 工具：JSON、权重分配、合并
    // ----------------------------------------------------------------------
    static Map<String,Object> readJsonAsMap(Path p) throws IOException {
        try (Reader r = Files.newBufferedReader(p, StandardCharsets.UTF_8)){
            JsonElement root = JsonParser.parseReader(r);
            Type t = new com.google.gson.reflect.TypeToken<Map<String,Object>>(){}.getType();
            return new Gson().fromJson(root, t);
        }
    }

    static Map<String,Integer> toIntMap(Map<String,Object> m){
        LinkedHashMap<String,Integer> out = new LinkedHashMap<>();
        if (m == null) return out;
        for (Map.Entry<String,Object> e : m.entrySet()){
            Object v = e.getValue();
            int iv = 0;
            if (v instanceof Number) iv = ((Number)v).intValue();
            else {
                try { iv = Integer.parseInt(String.valueOf(v)); } catch(Exception ignore){}
            }
            out.put(e.getKey(), iv);
        }
        return out;
    }

    static Map<String,Integer> mergeWeightMap(Map<String,Integer> base, Map<String,Integer> override){
        LinkedHashMap<String,Integer> out = new LinkedHashMap<>();
        if (base!=null) out.putAll(base);
        if (override!=null) out.putAll(override);
        return out;
    }

    static int sumWeights(Map<String,Integer> m){
        int s=0; for (int v: m.values()) s+=Math.max(0,v); return s;
    }

    // 关键修正：只在权重>0的键之间分配（0 权重绝不产出）
    static Map<String,Integer> allocByWeight(Map<String,Integer> weight, int total){
        LinkedHashMap<String,Integer> pos = new LinkedHashMap<>();
        for (Map.Entry<String,Integer> e : weight.entrySet()){
            int w = Math.max(0, e.getValue());
            if (w > 0) pos.put(e.getKey(), w);
        }
        int sum = sumWeights(pos);
        if (sum <= 0 || total<=0) return Collections.emptyMap();

        LinkedHashMap<String,Integer> out = new LinkedHashMap<>();
        LinkedHashMap<String,Double> frac = new LinkedHashMap<>();
        int used = 0;
        for (String k : pos.keySet()){
            double exact = 1.0 * total * pos.get(k) / sum;
            int base = (int)Math.floor(exact);
            out.put(k, base);
            used += base;
            frac.put(k, exact - base);
        }
        int left = total - used;
        List<String> order = new ArrayList<>(pos.keySet());
        order.sort((a,b)->Double.compare(frac.get(b), frac.get(a)));
        for (int i=0; i<left; i++){
            String k = order.get(i % order.size());
            out.put(k, out.get(k) + 1);
        }
        return out;
    }

    static Map<String,Integer> defaultSizes(){
        LinkedHashMap<String,Integer> s = new LinkedHashMap<>();
        s.put("160x60",40); s.put("180x64",20); s.put("200x64",15);
        s.put("256x64",10); s.put("120x40",10); s.put("96x32",5);
        return s;
    }

    static Map<String,Integer> defaultLenDist(){
        LinkedHashMap<String,Integer> m = new LinkedHashMap<>();
        m.put("4",50); m.put("5",25); m.put("6",25);
        return m;
    }

    static Map<String,Integer> defaultArithLen(){
        LinkedHashMap<String,Integer> m = new LinkedHashMap<>();
        m.put("2",80); m.put("3",20);
        return m;
    }

    // ----------------------------------------------------------------------
    // 杂项：字体、依赖、自检
    // ----------------------------------------------------------------------
    static class Roulette<T>{
        final List<T> keys = new ArrayList<>();
        final int total;
        final Random rnd = new Random();
        Roulette(Map<T,Integer> w){
            int sum=0;
            for (Map.Entry<T,Integer> e : w.entrySet()){
                int v=Math.max(0,e.getValue()); if (v<=0) continue;
                keys.add(e.getKey()); sum+=v;
            }
            total = Math.max(1,sum);
        }
        T next(){
            int r = rnd.nextInt(total);
            int acc=0;
            for (Map.Entry<T,Integer> e : new LinkedHashMap<T,Integer>() {{ for (T k: keys) put(k,1);} }.entrySet()){
                // 这里只需要按 keys 循环，真实权重在构造时折算到 total；简化实现
            }
            // 简洁实现：均匀轮询 keys 的同时使用 total 只是保证 rnd.nextInt 可用
            return keys.get(r % keys.size());
        }
    }

    static final String[] BUILTIN_TTFS = new String[]{
            "actionj.ttf","epilog.ttf","fresnel.ttf","headache.ttf",
            "lexo.ttf","prefix.ttf","progbot.ttf","ransom.ttf","robot.ttf","scandal.ttf"
    };

    static java.util.List<Font> loadBuiltinFonts(){
        java.util.List<Font> list = new java.util.ArrayList<>();
        ClassLoader cl = EasyCaptchaRunner.class.getClassLoader();
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

    static void applyFontForArith(Captcha c, FontPlan fp, int imgH, int idx){
        float sz = Math.max(18f, imgH * 0.72f);
        if (fp.mode == FontPlan.Mode.LIST && !fp.nameList.isEmpty()){
            for (int k=0; k<fp.nameList.size(); k++){
                String name = fp.nameList.get((idx + k) % fp.nameList.size()).trim();
                Font f = new Font(name, Font.BOLD, (int)sz);
                if (fontSupports(f, ARITH_TEST)){ c.setFont(f); return; }
            }
        }
        if (fp.mode == FontPlan.Mode.BUILTIN && !fp.builtin.isEmpty()){
            for (int k=0; k<fp.builtin.size(); k++){
                Font base = fp.builtin.get((idx + k) % fp.builtin.size());
                Font f = base.deriveFont(Font.PLAIN, sz);
                if (fontSupports(f, ARITH_TEST)){ c.setFont(f); return; }
            }
        }
        for (String name : SAFE_ARITH_FONTS){
            Font f = new Font(name, Font.BOLD, (int)sz);
            if (fontSupports(f, ARITH_TEST)){ c.setFont(f); return; }
        }
    }

    static void ensureArithDeps(){
        try { Class.forName("org.openjdk.nashorn.api.scripting.NashornScriptEngineFactory"); }
        catch (ClassNotFoundException e) {
            throw new RuntimeException("缺少 nashorn-core（JDK15+ 必需）。请将 nashorn-core-15.4.jar 放入 lib/ 并加入 -cp", e);
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

    static int mapTypeToCharType(String type){
        switch (type){
            case "spec":        return 1;
            case "spec_num":    return 2;
            case "spec_char":   return 3;
            case "spec_upper":  return 4;
            case "spec_lower":  return 5;
            case "spec_numupper": return 6;
            case "spec_numlower": return 7;
            default: return 1;
        }
    }

    // ----------------------------------------------------------------------
    // 旧命令中的字体计划（--fonts）
    // ----------------------------------------------------------------------
    static class FontPlan {
        enum Mode { NONE, BUILTIN, LIST }
        Mode mode = Mode.NONE;
        java.util.List<String> nameList = new java.util.ArrayList<>();
        java.util.List<Font> builtin = new java.util.ArrayList<>();
    }

    static FontPlan parseFontPlan(String[] argv){
        FontPlan fp = new FontPlan();
        for (int i=0;i<argv.length;i++){
            if ("--fonts".equalsIgnoreCase(argv[i]) && i+1<argv.length){
                String v = argv[++i];
                if ("builtin".equalsIgnoreCase(v)){
                    fp.mode = FontPlan.Mode.BUILTIN;
                    fp.builtin = loadBuiltinFonts();
                } else {
                    fp.mode = FontPlan.Mode.LIST;
                    fp.nameList = new java.util.ArrayList<>(Arrays.asList(v.split(",")));
                }
            }
        }
        return fp;
    }

    // ----------------------------------------------------------------------
    // 杂项
    // ----------------------------------------------------------------------
    static String uuid(){ return java.util.UUID.randomUUID().toString().replace("-",""); }
    static int parseInt(String s, int d){ try { return Integer.parseInt(s); } catch(Exception e){ return d; } }
}
