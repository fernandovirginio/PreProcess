package br.preprocess.core;

import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author Fernando
 */
public class KNearestNeighbour implements Classifier {
    
    private final IBk knn;
    private Evaluation evl;
    private Instances dsFull;
    private Instances dsTrain;
    private Instances dsTest;
    private boolean crossV;
    private int k;
    private boolean weight;
    
    public KNearestNeighbour(int k, boolean weight) {
        this.knn = new IBk();
        this.k = k;
        this.weight = weight;
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
            String w = (this.weight)?"-I ":"";
            this.knn.setOptions(Utils.splitOptions("-K "+this.k+" -W 0 "+w+"-A "
                    + "\"weka.core.neighboursearch.LinearNNSearch -A "
                    + "\\\"weka.core.EuclideanDistance -D -R first-last\\\"\""));
            this.knn.buildClassifier(this.dsTrain);
            
            if (this.crossV) {
                this.evl = new Evaluation(this.dsFull);
                this.evl.crossValidateModel(this.knn, this.dsFull, 10, new Random(0));
            } else {
                this.evl = new Evaluation(this.dsTrain);
                this.evl.evaluateModel(this.knn, this.dsTest);
            }
            return true;
        } catch (Exception ex) {
            return false;
        }
    }
    
    @Override
    public double[][] improve(int threshold) {
        double[][] results = new double[2][threshold];
        int best_k = 1;
        boolean best_w = false;
        double lower_err = 1;
        for (int kTest = 1; kTest <= threshold; kTest++) {
            boolean wTest = false;
            for (int i = 0; i < 2; i++) {
                this.k = kTest;
                this.weight = wTest;
                this.execute();

                if (this.weight)
                    results[1][kTest-1] = this.evl.pctCorrect()/100;
                else
                    results[0][kTest-1] = this.evl.pctCorrect()/100;
                
                if (this.evl.errorRate() < lower_err) {
                    best_k = kTest;
                    best_w = wTest;
                    lower_err = this.evl.errorRate();
                }
                wTest = true;
            }
        }
        this.k = best_k;
        this.weight = best_w;
        this.execute();
        return results;
    }
    
    @Override
    public Evaluation getEvl() {
        return evl;
    }

    @Override
    public String getWekaParameters() {
        return "weka.classifiers.lazy.IBk -- -K "+this.k+" -W 0 "+((this.weight)?"-I ":"")+"-A "
                + "\"weka.core.neighboursearch.LinearNNSearch -A "
                + "\\\"weka.core.EuclideanDistance -D -R first-last\\\"\"";
    }
    
    public void setK(int k) {
        this.k = k;
    }

    public void setWeight(boolean weight) {
        this.weight = weight;
    }

    public int getK() {
        return k;
    }

    public boolean isWeight() {
        return weight;
    }

    public void setCrossV(boolean crossV) {
        this.crossV = crossV;
    }
    
}
