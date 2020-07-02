package sample;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.layout.Pane;
import javafx.stage.Stage;


public class Main extends Application {


    private Button btn1 = new Button("Test 1 - Credit");
    private Button btn2 = new Button("Test 2 - Glass");
    private Button btn3 = new Button("Test 3 - Covid_19");
    Pane r=new Pane();

    @Override
    public void start(Stage primaryStage) throws Exception{
        //Parent root = FXMLLoader.load(getClass().getResource("sample.fxml"));
        primaryStage.setTitle("WekaApi");
        r.getChildren().add(btn1);
        btn1.setPrefSize(100,50);
        btn2.setPrefSize(100,50);
        btn3.setPrefSize(100,50);
        btn1.setLayoutX(50);
        btn1.setLayoutY(25);
        btn2.setLayoutX(50);
        btn2.setLayoutY(100);
        btn3.setLayoutX(50);
        btn3.setLayoutY(175);
        r.getChildren().add(btn2);
        r.getChildren().add(btn3);


        primaryStage.setScene(new Scene(r, 300, 250));



        primaryStage.show();


        btn1.setOnAction(actionEvent ->  {
            Weka prediction_credits=new Weka("src/german_credit.arff");
            try {
                prediction_credits.buildingModelPredicted();
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
        btn2.setOnAction(actionEvent ->  {
            Weka glass=new Weka("src/glass.arff");
            try {
                glass.buildingModel();
            } catch (Exception e) {
                e.printStackTrace();
            }
        });

        btn3.setOnAction(actionEvent ->  {
            Weka covid=new Weka("src/covid_19.csv");
            try {
                covid.buildingModel();
            } catch (Exception e) {
                e.printStackTrace();
            }
        });


}


    public static void main(String[] args) throws Exception {

        launch(args);


    }
}
