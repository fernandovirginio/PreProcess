package br.preprocess.core;

import weka.classifiers.Evaluation;

/**
 *
 * @author Fernando
 */
public interface Classifier {

    public void setDataset(Dataset ds, double splitRatio);
    public boolean execute();
    public double[][] improve(int threshold);
    public Evaluation getEvl();
    public String getWekaParameters();
    
}