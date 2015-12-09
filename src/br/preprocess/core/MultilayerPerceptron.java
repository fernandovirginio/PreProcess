package br.preprocess.core;

import java.util.Random;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author Fernando
 */
public class MultilayerPerceptron implements Classifier {
    
    private final weka.classifiers.functions.MultilayerPerceptron mlp;
    private Evaluation evl;
    private Instances dsFull;
    private Instances dsTrain;
    private Instances dsTest;
    private boolean crossV;
    private int neuro;
    private double learnRate;
    private int epochs;
    private int seed;
    
    public MultilayerPerceptron(int neuro, double learnRate, int epochs, int seed) {
        this.mlp = new weka.classifiers.functions.MultilayerPerceptron();
        this.neuro = neuro;
        this.learnRate = learnRate;
        this.epochs = epochs;
        this.seed = seed;
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
            this.mlp.setOptions(Utils.splitOptions("-L "+this.learnRate+" -M 0.0"
                    + " -N "+this.epochs+" -V 0 -S "+this.seed+" -E 20 -H "+this.neuro));
            this.mlp.buildClassifier(this.dsTrain);
            
            if (this.crossV) {
                this.evl = new Evaluation(this.dsFull);
                this.evl.crossValidateModel(this.mlp, this.dsFull, 10, new Random(0));
            } else {
                this.evl = new Evaluation(this.dsTrain);
                this.evl.evaluateModel(this.mlp, this.dsTest);
            }
            return true;
        } catch (Exception ex) {
            return false;
        }
    }
    
    @Override
    public double[][] improve(int threshold) {
        double[][] results = new double[1][threshold];
        int best_neuro = 1;
        double best_lrate = 0.01;
        int best_epochs = 100;
        double lower_err = 1;
        for (int nTest = 1; nTest <= threshold; nTest++) {
            double local_err = 1;
            for (double lrTest = 0.05; lrTest <= 0.3; lrTest += 0.05) {
                for (int epoTest = 100; epoTest <= 1000; epoTest += 100) {
                    this.neuro = nTest;
                    this.learnRate = lrTest;
                    this.epochs = epoTest;

                    this.execute();

                    if (this.evl.errorRate() < local_err) {
                        results[0][nTest-1] = this.evl.pctCorrect()/100;
                        local_err = this.evl.errorRate();
                    }
                    if (this.evl.errorRate() < lower_err) {
                        best_neuro = nTest;
                        best_lrate = lrTest;
                        best_epochs = epoTest;
                        lower_err = this.evl.errorRate();
                    }
                }
            }
        }
        this.neuro = best_neuro;
        this.learnRate = best_lrate;
        this.epochs = best_epochs;
        this.execute();
        return results;
    }
    
    @Override
    public Evaluation getEvl() {
        return evl;
    }

    @Override
    public String getWekaParameters() {
        return "weka.classifiers.functions.MultilayerPerceptron -- -L "+this.learnRate+" -M 0.0"
                + " -N "+this.epochs+" -V 0 -S 0 -E 20 -H "+this.neuro;
    }

    public void setNeuro(int neuro) {
        this.neuro = neuro;
    }

    public void setLearnRate(double learnRate) {
        this.learnRate = learnRate;
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    public void setSeed(int seed) {
        this.seed = seed;
    }
    
    public int getNeuro() {
        return neuro;
    }

    public double getLearnRate() {
        return learnRate;
    }

    public int getEpochs() {
        return epochs;
    }

    public void setCrossV(boolean crossV) {
        this.crossV = crossV;
    }
    
}
