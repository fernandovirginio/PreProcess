package br.preprocess.core;

import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.Utils;

import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.meta.Bagging;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RandomSubset;

public class BaggingSelect extends Bagging {

    static final long serialVersionUID = -505879962237199703L;
    protected double m_reductRatio;
    protected final RandomSubset m_filter = new RandomSubset();
    protected Instances[] m_ClassifiersDataset;
    
    public Instances reduceInstances(Instances data, int iteration) {
        Instances bagData = data;
        try {
            if (m_reductRatio > 0) {
                m_filter.setOptions(Utils.splitOptions("-N "+(1-m_reductRatio)+" -S "+iteration));
                m_filter.setInputFormat(data);
                bagData = Filter.useFilter(data, m_filter);
                m_ClassifiersDataset[iteration] = bagData;
            }
        } catch (Exception e) {}
        return bagData;
    }
    
    public Instance reduceInstance(Instance instance, int iteration) {
        if (m_reductRatio > 0) {
            Instance instanceAux = new DenseInstance(instance);
            boolean remove;
            for (int i = instance.numAttributes()-1; i >= 0; i--) {
                remove = true;
                for (int j = 0; j < m_ClassifiersDataset[iteration].numAttributes(); j++) {
                    Attribute a = m_ClassifiersDataset[iteration].attribute(j);
                    if (instance.attribute(i).name().equals(a.name()))
                        remove = false;
                }
                if (remove)
                    instanceAux.deleteAttributeAt(i);
            }
            instanceAux.setDataset(m_ClassifiersDataset[iteration]);
            return instanceAux;
        }
        return instance;
    }
    
    @Override
    public void setOptions(String[] options) throws Exception {
        String reductRatio = Utils.getOption('R', options);
        m_reductRatio = Double.parseDouble(reductRatio);
        super.setOptions(options);
    }
    
    @Override
    protected synchronized Instances getTrainingSet(int iteration) throws Exception {
        int bagSize = m_data.numInstances() * m_BagSizePercent / 100;
        Instances bagData;
        Random r = new Random(m_Seed + iteration);

        if (m_CalcOutOfBag) {
            m_inBag[iteration] = new boolean[m_data.numInstances()];
            bagData = m_data.resampleWithWeights(r, m_inBag[iteration]);
        } else {
            bagData = m_data.resampleWithWeights(r);
            if (bagSize < m_data.numInstances()) {
                bagData.randomize(r);
                Instances newBagData = new Instances(bagData, 0, bagSize);
                bagData = newBagData;
            }
        }
        
        // redução horizontal utilizando filtro randômico
        bagData = reduceInstances(bagData, iteration);
    
        return bagData;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        getCapabilities().testWithFail(data);

        m_data = new Instances(data);
        m_data.deleteWithMissingClass();

        if (m_CalcOutOfBag && (m_BagSizePercent != 100)) {
          throw new IllegalArgumentException("Bag size needs to be 100% if " +
                                             "out-of-bag error is to be calculated!");
        }

        m_random = new Random(m_Seed);
        m_Classifiers = SingleClassifierEnhancer.makeCopies(m_Classifier, m_NumIterations);
        m_ClassifiersDataset = new Instances[m_Classifiers.length];

        m_inBag = null;
        if (m_CalcOutOfBag)
            m_inBag = new boolean[m_Classifiers.length][];

        for (Classifier c : m_Classifiers) {
            if (m_Classifier instanceof Randomizable) {
                ((Randomizable) c).setSeed(m_random.nextInt());    
            }
        }

        buildClassifiers();

        // calc OOB error?
        if (getCalcOutOfBag()) {
            double outOfBagCount = 0.0;
            double errorSum = 0.0;
            boolean numeric = m_data.classAttribute().isNumeric();

            for (int i = 0; i < m_data.numInstances(); i++) {
                double vote;
                double[] votes;
                if (numeric)
                    votes = new double[1];
                else
                    votes = new double[m_data.numClasses()];

                // determine predictions for instance
                int voteCount = 0;
                
                for (int j = 0; j < m_Classifiers.length; j++) {
                    // redução das instâncias de acordo com a iteração
                    Instance instance = reduceInstance(m_data.instance(i), j);

                    if (m_inBag[j][i])
                        continue;

                    voteCount++;
                    if (numeric) {
                        votes[0] += m_Classifiers[j].classifyInstance(instance);
                    } else {
                        double[] newProbs = m_Classifiers[j].distributionForInstance(instance);
                        // average the probability estimates
                        for (int k = 0; k < newProbs.length; k++) {
                            votes[k] += newProbs[k];
                        }
                    }
                }

                // "vote"
                if (numeric) {
                    vote = votes[0];
                    if (voteCount > 0)
                        vote  /= voteCount;
                } else {
                    if (!Utils.eq(Utils.sum(votes), 0)) {            
                        Utils.normalize(votes);
                    }
                    vote = Utils.maxIndex(votes);   // predicted class
                }
              
                // error for instance
                outOfBagCount += m_data.instance(i).weight();
                if (numeric) {
                    errorSum += StrictMath.abs(vote - m_data.instance(i).classValue()) 
                    * m_data.instance(i).weight();
                } else {
                    if (vote != m_data.instance(i).classValue())
                        errorSum += m_data.instance(i).weight();
                }
            }
            m_OutOfBagError = errorSum / outOfBagCount;
        } else {
            m_OutOfBagError = 0;
        }
        m_data = null;
    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double [] sums = new double [instance.numClasses()], newProbs;
    
        for (int i = 0; i < m_NumIterations; i++) {
            // redução das instâncias para adaptação aos datasets reduzidos
            Instance instanceAux = reduceInstance(instance, i);
            
            if (instanceAux.classAttribute().isNumeric() == true) {
                sums[0] += m_Classifiers[i].classifyInstance(instanceAux);
            } else {
                newProbs = m_Classifiers[i].distributionForInstance(instanceAux);
                for (int j = 0; j < newProbs.length; j++)
                    sums[j] += newProbs[j];
            }
        }
        
        if (instance.classAttribute().isNumeric() == true) {
            sums[0] /= (double)m_NumIterations;
            return sums;
        } else if (Utils.eq(Utils.sum(sums), 0)) {
            return sums;
        } else {
            Utils.normalize(sums);
            return sums;
        }
    }

}