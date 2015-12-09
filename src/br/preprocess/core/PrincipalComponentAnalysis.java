package br.preprocess.core;

import weka.attributeSelection.PrincipalComponents;
import weka.core.Utils;

/**
 *
 * @author Fernando
 */
public class PrincipalComponentAnalysis {
    
    private final PrincipalComponents pca;
    private Dataset ds;
    private double varCovered;
    private int numComponents;
    private String info;
    
    
    public PrincipalComponentAnalysis(double varCovered) {
        this.pca = new PrincipalComponents();
        this.varCovered = varCovered;
    }
    
    public PrincipalComponentAnalysis() {
        this.pca = new PrincipalComponents();
    }

    public boolean reduceDataset() {
        try {
            this.pca.setOptions(Utils.splitOptions("-R "+this.varCovered+" -A 2 -M -1"));
            this.pca.buildEvaluator(this.ds.getDataset());

            this.ds.setDataset(this.pca.transformedData(this.ds.getDataset()));
            this.numComponents = this.ds.getNumAttr()-1;
            String stat = "Componentes principais";
            stat += "("+this.numComponents+"):\n";
            for (int i = 0; i < this.numComponents; i++) {
                stat += "PC"+(i+1)+" "+this.ds.getDataset().attribute(i).name()+"\n";
            }
            this.info = stat;
            
            return true;
        } catch (Exception ex) {
            return false;
        }
    }
    
    public void setVarCovered(double varCovered) {
        this.varCovered = varCovered;
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
    
    public int getNumComponents() {
        return this.numComponents;
    }
    
    public String getComponentNameAt(int index) {
        return this.ds.getDataset().attribute(index).name();
    }
    
}
