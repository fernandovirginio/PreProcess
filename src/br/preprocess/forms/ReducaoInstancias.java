/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package br.preprocess.forms;

import br.preprocess.App;
import br.preprocess.core.Chart;
import br.preprocess.core.Dataset;
import br.preprocess.core.KNearestNeighbour;
import br.preprocess.utils.Utils;
import javax.swing.JOptionPane;

/**
 *
 * @author Fernando
 */
public class ReducaoInstancias extends javax.swing.JDialog {

    private final App parent;
    private final Chart ch;
    private final Dataset ds;
    private final KNearestNeighbour knn;
    private int correl;
    
    private static final int SELEC_RANDOM = 1;
    
    /**
     * Creates new form WindowModel
     * @param parent
     * @param modal
     */
    public ReducaoInstancias(java.awt.Frame parent, boolean modal) {
        super(parent, modal);
        initComponents();
        this.parent = (App) parent;
        this.setLocation(parent.getLocation());
        this.ch = new Chart(this.redChart.getWidth(), this.redChart.getHeight(),
                    "Red. Instâncias - Desempenho", "Taxa de Redução", "Taxa de Acerto", Chart.CHART_LINE, true);
        this.ch.addXYSerie("Redução");
        Utils.addPlot(this.redChart, ch);
        this.ds = this.parent.getDsManipulavel().clonar();
        this.knn = this.parent.getKnn();
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jPanel1 = new javax.swing.JPanel();
        redChart = new javax.swing.JPanel();
        jPanel2 = new javax.swing.JPanel();
        jLabel2 = new javax.swing.JLabel();
        jLabel3 = new javax.swing.JLabel();
        redLimiar = new javax.swing.JSpinner();
        redRand = new javax.swing.JButton();
        redAplic = new javax.swing.JButton();
        redStat = new javax.swing.JLabel();

        setDefaultCloseOperation(javax.swing.WindowConstants.DISPOSE_ON_CLOSE);
        setTitle(" Redução de Instâncias");
        setIconImage(null);
        setResizable(false);

        jPanel1.setBackground(new java.awt.Color(255, 255, 255));

        redChart.setBackground(new java.awt.Color(255, 255, 255));

        javax.swing.GroupLayout redChartLayout = new javax.swing.GroupLayout(redChart);
        redChart.setLayout(redChartLayout);
        redChartLayout.setHorizontalGroup(
            redChartLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 354, Short.MAX_VALUE)
        );
        redChartLayout.setVerticalGroup(
            redChartLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 208, Short.MAX_VALUE)
        );

        jPanel2.setBackground(new java.awt.Color(255, 255, 255));
        jPanel2.setBorder(javax.swing.BorderFactory.createTitledBorder("Configuração"));

        jLabel2.setText("<html>\n<p>\nA avaliação de desempenho será realizada através do <strong>k-NN</strong> com a <strong>configuração atual</strong>.\n</p>\n</html>");
        jLabel2.setVerticalAlignment(javax.swing.SwingConstants.TOP);

        jLabel3.setText("Limiar de redução:");

        redLimiar.setModel(new javax.swing.SpinnerNumberModel(0.1d, 0.1d, 0.9d, 0.1d));

        redRand.setText("Utilizar seleção randômica");
        redRand.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                redRandActionPerformed(evt);
            }
        });

        redAplic.setFont(new java.awt.Font("Tahoma", 1, 11)); // NOI18N
        redAplic.setText("Aplicar Redução");
        redAplic.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                redAplicActionPerformed(evt);
            }
        });

        redStat.setFont(new java.awt.Font("Tahoma", 0, 10)); // NOI18N
        redStat.setText("<html></html>");
        redStat.setVerticalAlignment(javax.swing.SwingConstants.TOP);

        javax.swing.GroupLayout jPanel2Layout = new javax.swing.GroupLayout(jPanel2);
        jPanel2.setLayout(jPanel2Layout);
        jPanel2Layout.setHorizontalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel2Layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jLabel2, javax.swing.GroupLayout.PREFERRED_SIZE, 0, Short.MAX_VALUE)
                    .addGroup(jPanel2Layout.createSequentialGroup()
                        .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(jPanel2Layout.createSequentialGroup()
                                .addComponent(redStat, javax.swing.GroupLayout.PREFERRED_SIZE, 195, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(redAplic))
                            .addGroup(jPanel2Layout.createSequentialGroup()
                                .addComponent(jLabel3)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(redLimiar, javax.swing.GroupLayout.PREFERRED_SIZE, 55, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addComponent(redRand, javax.swing.GroupLayout.PREFERRED_SIZE, 167, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addGap(0, 0, Short.MAX_VALUE)))
                .addContainerGap())
        );
        jPanel2Layout.setVerticalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel2Layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jLabel2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel3)
                    .addComponent(redLimiar, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(redRand)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                    .addComponent(redAplic)
                    .addComponent(redStat, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(redChart, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jPanel2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(redChart, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jPanel2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(24, Short.MAX_VALUE))
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void redRandActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_redRandActionPerformed
        Dataset dsLocal = null;
        double split = this.parent.getSplitRatio();
        double threshold = (double) this.redLimiar.getValue();
        double best_pct = 0, best_lim = threshold;
        this.ch.clear();
        for (double i = 0; i <= threshold; i+= 0.1) {
            dsLocal = this.ds.clonar();
            dsLocal.reduceInstRandom(i);
            this.knn.setDataset(dsLocal, split);
            if (this.knn.execute())
                if (this.knn.getEvl().pctCorrect()/100 >= best_pct && i != 0) {
                    best_pct = this.knn.getEvl().pctCorrect()/100;
                    best_lim = i;
                }
            this.ch.update(i, this.knn.getEvl().pctCorrect()/100, 0);
        }
        this.redLimiar.setValue(best_lim);
        this.redStat.setText("<html><p>Ao aplicar você estará utilizando a seleção"
                + " randõmica de instâncias.</p></html>");
        this.correl = ReducaoInstancias.SELEC_RANDOM;
        Utils.addPlot(this.redChart, this.ch);
    }//GEN-LAST:event_redRandActionPerformed

    private void redAplicActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_redAplicActionPerformed
        if (this.correl != 0) {
            this.ds.reduceInstRandom(((double) this.redLimiar.getValue()));
            this.parent.setDsManipulavel(this.ds);
            this.parent.insertLog("Reduziu as instâncias da base de dados "+this.ds.getDatasetName()
                    + " utilizando seleção randômica.");
            this.parent.updateAll();
            this.dispose();
        } else {
            JOptionPane.showMessageDialog(this, "Avalie o método de redução randômica primeiro.");
        }
    }//GEN-LAST:event_redAplicActionPerformed
   
    public void open() {
        this.setVisible(true);
    }    
    
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JPanel jPanel2;
    private javax.swing.JButton redAplic;
    private javax.swing.JPanel redChart;
    private javax.swing.JSpinner redLimiar;
    private javax.swing.JButton redRand;
    private javax.swing.JLabel redStat;
    // End of variables declaration//GEN-END:variables
}
