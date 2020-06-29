package sample;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;



public class Weka {
    private Instances dataTesting;
    private Instances dataTraining;

    Weka(String location_training,String location_testing) {
        try {
           ConverterUtils.DataSource dataSourceTraining = new ConverterUtils.DataSource(location_training);
            ConverterUtils.DataSource dataSourceTesting = new ConverterUtils.DataSource(location_testing);
            dataTesting=dataSourceTesting.getDataSet();
            dataTraining=dataSourceTraining.getDataSet();
            dataTraining.setClassIndex(dataTraining.numAttributes()-1);
            dataTesting.setClassIndex(dataTesting.numAttributes()-1);

        } catch (Exception ex) {
            System.out.println("The dataset is incorrect or it can't be founded" + ex);
        }
        System.out.println("The dataset was loaded correctly");
    }

    void buildingModel() throws Exception {
        //System.out.print(dataTraining);
        String[] options = new String[1];
        options[0] = "-U";
        J48 cls= new J48();
        cls.setOptions(options);
        cls.buildClassifier(dataTraining);
        Evaluation eval=new Evaluation(dataTesting);
        eval.evaluateModel(cls,dataTraining);

        FastVector pred= (FastVector) eval.predictions();

        pred.appendElements(eval.predictions());
        System.out.println("test");
        Object[] vec=eval.predictions().toArray();
        for(int i=0;i<vec.length;i++)
        {
            System.out.println(vec[i]);


        }





        System.out.println(eval.toSummaryString("\nResult\n======\n",false));

    }





}
