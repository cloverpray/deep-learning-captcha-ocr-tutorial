// FontLister.java
import java.awt.*;
import java.util.*;

public class FontLister {
    static final String SAMPLE_ASCII = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    static final String SAMPLE_ARITH = "0123456789+-×÷=？";
    static final String SAMPLE_CN    = "验证码中文测试一二三四五六七八九零";

    static boolean supports(Font f, String s){
        try { return f.canDisplayUpTo(s) == -1; } catch (Throwable t) { return false; }
    }

    public static void main(String[] args){
        Locale loc = Locale.CHINA; // 也可改成 Locale.ROOT
        String[] names = GraphicsEnvironment.getLocalGraphicsEnvironment()
                .getAvailableFontFamilyNames(loc);
        Arrays.sort(names, String.CASE_INSENSITIVE_ORDER);

        System.out.println("本机可用字体数量 = " + names.length);
        System.out.println("name\tASCII\tARITH\tCN");

        for (String name : names){
            // 选择一个较大的字号来测试字形可显示性
            Font f = new Font(name, Font.PLAIN, 32);
            String a = supports(f, SAMPLE_ASCII) ? "Y" : "-";
            String r = supports(f, SAMPLE_ARITH) ? "Y" : "-";
            String c = supports(f, SAMPLE_CN)    ? "Y" : "-";
            System.out.println(name + "\t" + a + "\t" + r + "\t" + c);
        }
    }
}
