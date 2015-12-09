package br.preprocess.core;

import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author Fernando
 */
public class HomogeneousEnsemble {
    
    private weka.classifiers.Classifier strategy;
    private int strategyNumber;
    private Evaluation evl;
    private Instances dsFull;
    private Instances dsTrain;
    private Instances dsTest;
    private boolean crossV;
    private int iterations;
    private double reductRatio;
    
    public static int BAGGING = 0;
    public static int BOOSTING = 1;
    public static int BAGGING_SELECTION = 2;
    
    public HomogeneousEnsemble(int strategy, int iterations) {
        this.strategyNumber = strategy;
        this.iterations = iterations;
        this.reductRatio = 0;
    }
    
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
    
    public boolean execute(String parameters) {
        try {
            if (this.strategyNumber == HomogeneousEnsemble.BAGGING){
                Bagging bag = new Bagging();
                bag.setOptions(Utils.splitOptions("-P 100 -S 1 -num-slots 4 -I "
                        + this.iterations+" -W "+parameters));
                this.strategy = bag;
            } else if (this.strategyNumber == HomogeneousEnsemble.BOOSTING) {
                AdaBoostM1 bos = new AdaBoostM1();
                bos.setOptions(Utils.splitOptions("-P 100 -S 1 -I "+this.iterations
                        + " -W "+parameters));
                this.strategy = bos;
            } else {
                BaggingSelect bagR = new BaggingSelect();
                bagR.setOptions(Utils.splitOptions("-P 100 -S 1 -R "+this.reductRatio
                        + " -num-slots 4 -I "+this.iterations
                        + " -W "+parameters));
                this.strategy = bagR;
            }
            
            if (this.crossV) {
                this.evl = new Evaluation(this.dsFull);
                this.evl.crossValidateModel(this.strategy, this.dsFull, 10, new Random(0));
            } else {
                this.strategy.buildClassifier(this.dsTrain);
                this.evl = new Evaluation(this.dsTrain);
                this.evl.evaluateModel(this.strategy, this.dsTest);
            }
            return true;
        } catch (Exception ex) {
            return false;
        }
    }

    public double[] improve(int[] iterationSet, String parameters) {
        double[] results = new double[iterationSet.length];
        int best_iterate = 0;
        double lower_err = 1;
        for (int i = 0; i <= iterationSet.length; i++) {
            this.iterations = iterationSet[i];
                        
            this.execute(parameters);
            
            if (this.evl.errorRate() < lower_err) {
                best_iterate = iterationSet[i];
                lower_err = this.evl.errorRate();
            }
        }
        this.iterations = best_iterate;
        this.execute(parameters);
        return results;
    }

    public Evaluation getEvl() {
        return evl;
    }
    
    public void setStrategyNumber(int strategyNumber) {
        this.strategyNumber = strategyNumber;
    }

    public void setIterations(int iterations) {
        this.iterations = iterations;
    }

    public void setReductRatio(double reductRatio) {
        this.reductRatio = reductRatio;
    }

    public int getStrategyNumber() {
        return strategyNumber;
    }

    public int getIterations() {
        return iterations;
    }

    public void setCrossV(boolean crossV) {
        this.crossV = crossV;
    }
    
}