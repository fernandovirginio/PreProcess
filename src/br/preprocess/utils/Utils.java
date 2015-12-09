package br.preprocess.utils;

import br.preprocess.core.Chart;
import javax.swing.GroupLayout;
import javax.swing.JPanel;

/**
 *
 * @author Fernando
 */
public class Utils {
    
    public static int[] InsertionSort( double[] num, int[]pos) {
        int j, i, key2;
        double key;
        
        for (j = 1; j < num.length; j++) {
            key = num[j];
            key2 = pos[j];
            for(i = j-1; (i >= 0) && (num[i] > key); i--) {
                num[i+1] = num[i];
                pos[i+1] = pos[i];
            }
            num[i+1] = key;
            pos[i+1] = key2;
        }
        
        return pos;
    }
    
    public static int[] InsertionSortReverse( double[] num, int[]pos) {
        int j, i, key2;
        double key;
        
        for (j = 1; j < num.length; j++) {
            key = num[j];
            key2 = pos[j];
            for(i = j-1; (i >= 0) && (num[i] < key); i--) {
                num[i+1] = num[i];
                pos[i+1] = pos[i];
            }
            num[i+1] = key;
            pos[i+1] = key2;
        }
        
        return pos;
    }
    
    public static double[] InsertionSortReverse( double[] num) {
        int j, i;
        double key;
        
        for (j = 1; j < num.length; j++) {
            key = num[j];
            for(i = j-1; (i >= 0) && (num[i] < key); i--) {
                num[i+1] = num[i];
            }
            num[i+1] = key;
        }
        
        return num;
    }
    
    public static void addPlot(JPanel target, Chart ch) {
        target.removeAll();
        GroupLayout mainPanelLayout = new GroupLayout(target);
        target.setLayout(mainPanelLayout);
        GroupLayout.SequentialGroup hGroup = mainPanelLayout.createSequentialGroup();
        hGroup.addGap(1, 1, 1);
        hGroup.addComponent(ch.getImage());
        mainPanelLayout.setHorizontalGroup(hGroup);
        GroupLayout.SequentialGroup vGroup = mainPanelLayout.createSequentialGroup();
        vGroup.addGap(1, 1, 1);
        vGroup.addComponent(ch.getImage());
        mainPanelLayout.setVerticalGroup(vGroup);
        ch.getImage().setVisible(true);
    }
    
    public static void printMatrix(double[][] mat) {
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[i].length; j++) {
                System.out.print(mat[i][j]+"\t");
            }
            System.out.print("\n");
        }
    }
    
    public static void printVector(double[] vec) {
        for (int i = 0; i < vec.length; i++) {
            System.out.print(vec[i]+"\t");
        }
        System.out.print("\n");
    }

    public static void printVector(int[] vec) {
        for (int i = 0; i < vec.length; i++) {
            System.out.print(vec[i]+"\t");
        }
        System.out.print("\n");
    }
    
}
