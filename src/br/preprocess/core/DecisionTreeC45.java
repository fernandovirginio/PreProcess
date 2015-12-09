package br.preprocess.core;

import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author Fernando
 */
public class DecisionTreeC45 implements Classifier {
    
    private final J48 tree;
    private Evaluation evl;
    private Instances dsFull;
    private Instances dsTrain;
    private Instances dsTest;
    private boolean crossV;
    private double cFactor;
    private boolean pruned;
    
    public DecisionTreeC45(double cFactor, boolean pruned) {
        this.tree = new J48();
        this.cFactor = cFactor;
        this.pruned = pruned;
        this.crossV = false;
    }

    @Override
    public void setDataset(Dataset ds, double splitRatio) {
        Instances data = ds.getDataset();
        int trainSize = (int) (splitRatio*data.numInstances());
        int testSize = data.numInstances()-trainSize;
        
        this.dsFull = new Instances(data);
        this.dsTrain = new Instances(data, 0, trainSize);
        this.dsTrain.randomize(new Random(0));
        this.dsTrain.setClassIndex(data.numAttributes()-1);
        this.dsTest = new Instances(data, trainSize, testSize);
        this.dsTest.setClassIndex(data.numAttributes()-1);
    }
    
    @Override
    public boolean execute() {
        try {
            this.tree.setOptions(Utils.splitOptions(((this.pruned)?"-C "+this.cFactor:"-U")+" -M 2"));
            this.tree.buildClassifier(this.dsTrain);
            
            if (this.crossV) {
                this.evl = new Evaluation(this.dsFull);
                this.evl.crossValidateModel(this.tree, this.dsFull, 10, new Random(0));
            } else {
                this.evl = new Evaluation(this.dsTrain);
                this.evl.evaluateModel(this.tree, this.dsTest);
            }
            return true;
        } catch (Exception ex) {
            return false;
        }
    }
    
    @Override
    public double[][] improve(int threshold) {
        double[][] results = new double[1][2];
        boolean best_pruned = false;
        double best_cfactor = 0.01;
        double lower_err = 1;
        for (int i = 0; i < 2; i++) {
            if (i == 1) {
                for (double cTest = 0.05; cTest < 0.5; cTest += 0.05) {
                    this.pruned = true;
                    this.cFactor = cTest;

                    this.execute();
                    
                    results[0][0] = this.evl.pctCorrect()/100;
                    if (this.evl.errorRate() < lower_err) {
                        best_pruned = true;
                        best_cfactor = cTest;
                        lower_err = this.evl.errorRate();
                    }
                }
            } else {
                this.pruned = false;
                
                this.execute();
                
                results[0][1] = this.evl.pctCorrect()/100;
                if (this.evl.errorRate() < lower_err) {
                    best_pruned = false;
                    lower_err = this.evl.errorRate();
                }
            }
        }
        this.pruned = best_pruned;
        this.cFactor = best_cfactor;
        this.execute();
        return results;
    }
    
    @Override
    public Evaluation getEvl() {
        return evl;
    }

    @Override
    public String getWekaParameters() {
        return "weka.classifiers.trees.J48 -- "+((this.pruned)?"-C "+this.cFactor:"-U")+" -M 2";
    }

    public double getcFactor() {
        return cFactor;
    }

    public boolean isPruned() {
        return pruned;
    }

    public void setcFactor(double cFactor) {
        this.cFactor = cFactor;
    }

    public void setPruned(boolean pruned) {
        this.pruned = pruned;
    }

    public void setCrossV(boolean crossV) {
        this.crossV = crossV;
    }
    
}
