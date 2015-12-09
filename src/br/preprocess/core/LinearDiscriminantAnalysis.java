package br.preprocess.core;

import weka.classifiers.functions.FLDA;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Fernando
 */
public class LinearDiscriminantAnalysis {
    
    private final FLDA lda;
    private Dataset ds;
    private int numAttrs;
    private String info;
    private double ratio;
    private double threshold;
    private int[] attrs;
    
    public LinearDiscriminantAnalysis() {
        this.lda = new FLDA();
    }

    public boolean reduceDataset() {
        try {
            int numReduct = (int) Math.ceil(this.ratio*this.ds.getNumAttr());
            this.lda.buildClassifier(this.ds.getDataset());
            double[] weights = new double[this.ds.getNumAttr()-1];
            this.attrs = new int[this.ds.getNumAttr()-1];
            int w = 0;
            
            String[] stats = this.lda.toString().split("\n");
            for (int i = 0; i < stats.length; i++) {
                if (i == 2 || i >= 6) {
                    String[] stat = stats[i].split(":");
                    if (i == 2) this.threshold = Double.parseDouble(stat[1]);
                    if (i >= 6) {
                        weights[w] = Double.parseDouble(stat[1]);
                        this.attrs[w] = w;
                        w++;
                    }
                }
            }
            this.attrs = br.preprocess.utils.Utils.InsertionSort(weights, this.attrs);
            this.info = "Ordem decrescente de importância por peso dos atributos:\n";
            for (int i = 0; i < this.attrs.length; i++)
                this.info += (this.attrs[i]+1)+"\t"+this.ds.getAttrsName()[this.attrs[i]]+"\n";
            this.info += "\n\n"+this.lda.toString();
            
            Remove rmv = new Remove();
            int[] attrC = new int[numReduct];
            for (int i = 0; i < numReduct; i++)
                attrC[i] = this.attrs[i];
            rmv.setAttributeIndicesArray(attrC);
            rmv.setInputFormat(this.ds.getDataset());
            this.ds.setDataset(Filter.useFilter(this.ds.getDataset(), rmv));
            
            return true;
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
            return false;
        }
    }

    public void setRatio(double ratio) {
        this.ratio = ratio;
    }
    
    public void setDataset(Dataset ds) {
        this.ds = ds;
    }
    
    public Dataset getDataset() {
        return this.ds;
    }
    
    public String getInfo() {
        return this.info;
    }
    
    public int getNumAttrs() {
        return this.numAttrs;
    }
    
    public String getComponentNameAt(int index) {
        return this.ds.getDataset().attribute(index).name();
    }
    
}
