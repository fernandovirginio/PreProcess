package br.preprocess.core;

import java.util.Random;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author Fernando
 */
public class NaiveBayes implements Classifier {
    
    private final weka.classifiers.bayes.NaiveBayes nb;
    private Evaluation evl;
    private Instances dsFull;
    private Instances dsTrain;
    private Instances dsTest;
    private boolean crossV;
    private boolean kernel;
    private boolean discretize;
    
    public NaiveBayes(boolean kernel, boolean discretize) {
        this.nb = new weka.classifiers.bayes.NaiveBayes();
        this.kernel = kernel;
        this.discretize = discretize;
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
            if (this.kernel) {
                this.nb.setOptions(Utils.splitOptions("-K"));
            } else if (this.discretize) {
                this.nb.setOptions(Utils.splitOptions("-D"));
            }
            
            this.nb.buildClassifier(this.dsTrain);
            
            if (this.crossV) {
                this.evl = new Evaluation(this.dsFull);
                this.evl.crossValidateModel(this.nb, this.dsFull, 10, new Random(0));
            } else {
                this.evl = new Evaluation(this.dsTrain);
                this.evl.evaluateModel(this.nb, this.dsTest);
            }
            return true;
        } catch (Exception ex) {
            return false;
        }
    }
    
    @Override
    public double[][] improve(int threshold) {
        double[][] results = new double[1][3];
        boolean best_kernel = false;
        boolean best_discretize = false;
        double lower_err = 1;
        for (int i = 0; i < 3; i++) {
            switch (i) {
                case 0:
                    this.kernel = false;
                    this.discretize = false;
                    break;
                case 1:
                    this.kernel = true;
                    this.discretize = false;
                    break;
                case 2:
                    this.kernel = false;
                    this.discretize = true;
                    break;
                default:
                    break;
            }
            this.execute();
            
            results[0][i] = this.evl.pctCorrect()/100;
            if (this.evl.errorRate() < lower_err) {
                best_kernel = this.kernel;
                best_discretize = this.discretize;
                lower_err = this.evl.errorRate();
            }
        }
        this.kernel = best_kernel;
        this.discretize = best_discretize;
        this.execute();
        return results;
    }
    
    @Override
    public Evaluation getEvl() {
        return evl;
    }

    @Override
    public String getWekaParameters() {
        String par = "";
        if (this.kernel) {
            par = " -- -K";
        } else if (this.discretize) {
            par = " -- -D";
        }
        return "weka.classifiers.bayes.NaiveBayes"+par;
    }

    public boolean isKernel() {
        return kernel;
    }

    public boolean isDiscretize() {
        return discretize;
    }

    public void setKernel(boolean kernel) {
        this.kernel = kernel;
    }

    public void setDiscretize(boolean discretize) {
        this.discretize = discretize;
    }

    public void setCrossV(boolean crossV) {
        this.crossV = crossV;
    }
    
}
