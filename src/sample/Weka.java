package sample;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;



public class Weka {
    //creating variable for DataSets
    private Instances dataTesting;
    private Instances dataTraining;

    Weka(String location_training,String location_testing) {
        try {

            //Loading Datasets
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
        //Options for model
        String[] options = new String[1];
        options[0] = "-U";
        //Creating model
        J48 cls= new J48();
        cls.setOptions(options);
        cls.buildClassifier(dataTraining);
        Evaluation eval=new Evaluation(dataTesting);
        eval.evaluateModel(cls,dataTraining);
        System.out.println("Actual || Predicted");
        //Creating prediction for a TestDataSet
        for(int i=0;i<dataTesting.numInstances();i++)
        {
            double actualClass=dataTesting.instance(i).classValue();
            String actual=dataTesting.classAttribute().value((int) actualClass);
            Instance newInst=dataTesting.instance(i);
            double predJ48=cls.classifyInstance(newInst);
            String predString=dataTesting.classAttribute().value((int) predJ48);
            //Printing prediction
            System.out.println(actual+"  "+predString);


        }



        //Printing results about model
        System.out.println(eval.toSummaryString("\nResult\n======\n",false));

    }





}
