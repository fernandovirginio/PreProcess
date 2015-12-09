package br.preprocess.core;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import weka.attributeSelection.PrincipalComponents;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.RandomSubset;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 *
 * @author Fernando
 */
public class Dataset implements Cloneable {
    
    private String path;
    private Instances dataset;
    private String datasetName;
    private int numClass;
    private int[] numInstacesClass;
    private int numAttr;
    private int numInstances;
    private int attrNumeric;
    private int attrCategorical;
    private int[] attrCorrelate;
    private String[] attrsName;
    private String[] attrsType;
    private String[] attrsEscala;
    private double[] attrsMean;
    private double[] attrsStDev;
    private double[] attrsVaria;
    private double[][] corMatrix;
    
    public Dataset(String path) {
        this.path = path;
    }
    
    public boolean reduceInstRandom(double ratio) {
        if (this.dataset != null) {
            RemovePercentage rand = new RemovePercentage();
            try {
                if (ratio > 0) {
                    rand.setOptions(Utils.splitOptions("-P "+(ratio*100)));
                    rand.setInputFormat(this.dataset);
                    this.dataset = Filter.useFilter(this.dataset, rand);

                    this.loadAllStats();
                }
                return true;
            } catch (Exception e) {}
            return false;
        } else return false;
    }
    
    public boolean reduceAttrRandom(double ratio) {
        if (this.dataset != null) {
            RandomSubset rand = new RandomSubset();
            try {
                if (ratio < 1) {
                    rand.setOptions(Utils.splitOptions("-N "+ratio+" -S 1"));
                    rand.setInputFormat(this.dataset);
                    this.dataset = Filter.useFilter(this.dataset, rand);

                    this.loadAllStats();
                }
                return true;
            } catch (Exception e) {}
            return false;
        } else return false;
    }
    
    public boolean reduceAttrCorrelate(double ratio) {
        if (this.dataset != null) {
            Remove corr = new Remove();
            try {
                if (ratio > 0) {
                    int threshold = (int) Math.ceil(ratio*(this.numAttr-1));
                    int[] attrC = new int[threshold];
                    for (int i = 0; i < threshold; i++)
                        attrC[i] = this.attrCorrelate[i];
                    corr.setAttributeIndicesArray(attrC);
                    corr.setInputFormat(this.dataset);
                    this.dataset = Filter.useFilter(this.dataset, corr);
                    
                    this.loadAllStats();
                }
                return true;
            } catch (Exception e){}
            return false;
        } else return false;
    }
    
    public boolean convertToNumeric() {
        if (this.dataset != null) {
            NominalToBinary conv = new NominalToBinary();
            try {
                conv.setOptions(Utils.splitOptions("-R first-last"));
                conv.setInputFormat(this.dataset);
                this.dataset = Filter.useFilter(this.dataset, conv);
                
                this.loadAllStats();
                
                return true;
            } catch (Exception ex) {}
            return false;
        } else return false;
    }
    
    public boolean normalize() {
        if (this.dataset != null) {
            Normalize n = new Normalize();
            try {
                n.setOptions(Utils.splitOptions("-S 1.0 -T 0.0"));
                n.setInputFormat(this.dataset);
                this.dataset = Filter.useFilter(this.dataset, n);
                
                this.loadAllStats();
                
                return true;
            } catch (Exception ex) {}
            return false;
        } else return false;
    }
    
    public boolean saveDataset(String path) {
        try {
            if (!path.contains(".arff"))
                path += ".arff";
            ArffSaver arff = new ArffSaver();
            File file = new File(path);
            
            arff.setFile(file);
            arff.setInstances(this.dataset);
            
            arff.writeBatch();
            return true;
        } catch (Exception e) {}
        return false;
    }
    
    public boolean loadDataset() {
        try {
            try (BufferedReader reader = new BufferedReader(new FileReader(this.path))) {
                Instances data = new Instances(reader);
                data.setClassIndex(data.numAttributes()-1);
                this.dataset = new Instances(data);
                this.datasetName = data.relationName();
                this.numClass = data.numClasses();
                this.numInstacesClass = new int[this.numClass];
                reader.close();
                
                this.loadAllStats();
                
                return true;
            }
        } catch (Exception ex) {}
        return false;
    }
    
    private void loadAllStats() {
        this.numAttr = this.dataset.numAttributes();
        this.numInstances = this.dataset.numInstances();
        
        this.attrCategorical = 0;
        this.attrNumeric = 0;
        this.attrsName = new String[this.numAttr];
        this.attrsType = new String[this.numAttr];
        this.attrsEscala = new String[this.numAttr];
        this.attrsMean = new double[this.numAttr];
        this.attrsStDev = new double[this.numAttr];
        this.attrsVaria = new double[this.numAttr];

        try {
            PrincipalComponents pca = new PrincipalComponents();
            pca.buildEvaluator(this.dataset);
            this.corMatrix = pca.getCorrelationMatrix();
        } catch (Exception ex) {}
        
        double[] maxCor = new double[this.corMatrix.length];
        int[] indCor = new int[this.corMatrix.length];
        double max = 0;
        for (int linha = 0; linha < this.corMatrix.length; linha++) {
            for (int coluna = 0; coluna < this.corMatrix[linha].length; coluna++) {
                if (this.corMatrix[linha][coluna] > max && linha != coluna)
                    max = this.corMatrix[linha][coluna];
            }
            maxCor[linha] = max;
            indCor[linha] = linha;
            max = 0;
        }
        this.attrCorrelate = br.preprocess.utils.Utils.InsertionSortReverse(maxCor, indCor);
        
        Instance inst;
        double min_val, max_val, mean, std_dev, var, quad_sum;
        double std_dev_i[] = new double[this.numInstances];

        for (int i = 0; i < this.numAttr; i++) {
            this.attrsName[i] = this.dataset.attribute(i).name();
            mean = 0;
            std_dev = 0;
            var = 0;
            min_val = Double.MAX_VALUE;
            max_val = 0;
            if (this.dataset.classIndex() == i) {
                for (int j = 0; j < this.numClass; j++)
                    this.numInstacesClass[j] = 0;
                for (int j = 0; j < this.numInstances; j++) {
                    inst = this.dataset.instance(j);
                    this.numInstacesClass[(int) inst.value(i)]++;
                }
            }
            if (this.dataset.attribute(i).isNumeric()) {
                this.attrNumeric++;
                this.attrsType[i] = "Numérico";
                for (int j = 0; j < this.numInstances; j++) {
                    inst = this.dataset.instance(j);
                    mean += inst.value(i);
                    std_dev_i[j] = inst.value(i);
                    if (inst.value(i) > max_val) max_val = inst.value(i);
                    if (inst.value(i) < min_val) min_val = inst.value(i);
                }
                mean /= this.numInstances;
                quad_sum = 0;
                for (int j = 0; j < this.numInstances; j++) {
                    quad_sum += Math.pow(std_dev_i[j]-mean, 2);
                }
                var = quad_sum/this.numInstances;
                std_dev = Math.sqrt(var);
            } else {
                this.attrCategorical++;
                this.attrsType[i] = "Categórico";
            }
            this.attrsEscala[i] = min_val+" - "+max_val;
            this.attrsMean[i] = mean;
            this.attrsStDev[i] = std_dev;
            this.attrsVaria[i] = var;
        }
    }
    
    public String getStats() {
        return this.dataset.toSummaryString();
    }

    public void setPath(String path) {
        this.path = path;
    }

    protected void setDataset(Instances ds) {
        this.dataset = ds;
        
        this.loadAllStats();
    }
    
    public int getNumClass() {
        return numClass;
    }

    public int[] getAttrCorrelate() {
        return attrCorrelate;
    }
    
    public int getNumAttr() {
        return numAttr;
    }

    public int getNumInstances() {
        return numInstances;
    }

    public int getAttrNumeric() {
        return attrNumeric;
    }

    public int getAttrCategorical() {
        return attrCategorical;
    }
    
    public Instances getDataset() {
        return dataset;
    }
    
    public String getDatasetName() {
        return datasetName;
    }
    
    public int getNumInstancesClass(int classe) {
        return numInstacesClass[classe];
    }
    
    public String[] getAttrsName() {
        return attrsName;
    }

    public String[] getAttrsType() {
        return attrsType;
    }

    public String[] getAttrsEscala() {
        return attrsEscala;
    }
    
    public double[] getAttrsMean() {
        return attrsMean;
    }

    public double[] getAttrsStDev() {
        return attrsStDev;
    }

    public double[] getAttrsVaria() {
        return attrsVaria;
    }

    public double[][] getCorMatrix() {
        return corMatrix;
    }
    
    public Dataset clonar() {
        try {
            return this.clone();
        } catch (CloneNotSupportedException ex) {}
        return null;
    }
    
    @Override
    public Dataset clone() throws CloneNotSupportedException {
        try {
            Dataset clone = (Dataset) super.clone();
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            try (ObjectOutputStream out = new ObjectOutputStream(bos)) {
                out.writeObject(this.dataset);
            }
            
            ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
            try (ObjectInputStream in = new ObjectInputStream(bis)) {
                clone.dataset = (Instances) in.readObject();
            }
            
            return clone;
        } catch (IOException | ClassNotFoundException ex) {}
        return null;
    }
    
}
