package sample;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;



public class Weka {
    //creating variable for DataSets

    private Instances dataTraining;


    Weka(String location_dataset)
    {
        try {

            //Loading Datasets
            ConverterUtils.DataSource dataSourceTraining = new ConverterUtils.DataSource(location_dataset);
            dataTraining=dataSourceTraining.getDataSet();
            dataTraining.setClassIndex(dataTraining.numAttributes()-1);
        } catch (Exception ex) {
            System.out.println("The dataset is incorrect or it can't be founded " + ex);
        }
        System.out.println("The dataset was loaded correctly");
    }




    void buildingModelPredicted() throws Exception {
        //Options for model
        String[] options = new String[1];
        options[0] = "-U";
        //Creating model
        RandomForest cls= new RandomForest();
        cls.setOptions(options);

        //spliting the dataset into
        int trainSize = (int) Math.round(dataTraining.numInstances() * 0.75);
        int testSize = dataTraining.numInstances() - trainSize;
        Instances train = new Instances(dataTraining, 0, trainSize);
        Instances test = new Instances(dataTraining, trainSize, testSize);


        long start=System.currentTimeMillis();
        cls.buildClassifier(train);




        long elapsedTime=System.currentTimeMillis()-start;
        Evaluation eval=new Evaluation(test);
        eval.evaluateModel(cls,test);
        System.out.println("Actual || Predicted");
        //Creating prediction for a TestDataSet
        int well_predicted=0;
        for(int i=0;i<test.numInstances();i++)
        {
            double actualClass=test.instance(i).classValue();
            String actual=test.classAttribute().value((int) actualClass);
            Instance newInst=test.instance(i);
            double predJ48=cls.classifyInstance(newInst);
            String predString=test.classAttribute().value((int) predJ48);
            if(actual.equals(predString)) well_predicted++;
            //Printing prediction
            System.out.println(actual+"  "+predString);

        }


        //Printing results about model
        System.out.println(eval.toSummaryString("\nResult\n======\n",false));
        System.out.println("Percent of correct predicted values: "+ (double)well_predicted/test.numInstances()*100 +"%" );
        System.out.println("Time takes to build the model: "+elapsedTime+" ms");
        System.out.println("Train - test " + train.size() + " " + test.size());

    }

    void buildingModel() throws Exception {
        //Options for model
        String[] options = new String[1];
        options[0] = "-U";
        //Creating model
        RandomForest cls= new RandomForest();
        cls.setOptions(options);

        long start=System.currentTimeMillis();
        cls.buildClassifier(dataTraining);
        long elapsedTime=System.currentTimeMillis()-start;
        Evaluation eval=new Evaluation(dataTraining);
        eval.evaluateModel(cls,dataTraining);


        //Printing results about model
        System.out.println(eval.toSummaryString("\nResult\n======\n",false));

        System.out.println("Time takes to build the model: "+elapsedTime+" ms");

    }





}
