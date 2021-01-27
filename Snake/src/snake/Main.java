package snake;

import ai.*;
import javafx.application.Application;
import static javafx.application.Application.launch;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.control.Slider;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;
import javafx.stage.Stage;

public class Main extends Application {

    public final Group ROOT = new Group();
    private static final boolean whileloop = false;
    private static final int gridWidth = 6;
    private static final int gridHeight = 6;
    private static final int gridUnitSize = 50;
    private static final int numberOfAgents = 1;

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage stage) throws Exception {
        if (!whileloop) {
            Scene scene = new Scene(ROOT, 400, 60);
            stage.setScene(scene);
            stage.setResizable(false);
            stage.sizeToScene();
            stage.show();
            Snake[] games = new Snake[numberOfAgents];
            for (int i = 0; i < numberOfAgents; i++) {
                Snake game = new Snake(gridWidth, gridHeight, gridUnitSize, whileloop);
                games[i] = game;
                insertGameIntoAgent(game);
                game.play();
            }
            ROOT.getChildren().add(gameSpeedSlider(games));
        } else {
            for (int i = 0; i < numberOfAgents; i++) {
                Snake game = new Snake(gridWidth, gridHeight, gridUnitSize, whileloop);
                insertGameIntoAgent(game);
                Thread t = new Thread(() -> game.play());
                t.setDaemon(true);
                t.start();
            }
        }
    }

    private void insertGameIntoAgent(Snake game) {
//        Naive naive = new Naive(game);
//        Genetic genetic = new Genetic(game);
//        REINFORCE pg = new REINFORCE(game);
//        ActorCritic ac = new ActorCritic(game);
//        ActorCriticTD acTD = new ActorCriticTD(game);
        PPO ppo = new PPO(game);
    }

    private VBox gameSpeedSlider(Snake[] games) {
        Text rate = new Text("1.0");
        Slider slider = new Slider();
        slider.setMin(1);
        slider.setMax(500);
        slider.setBlockIncrement(5);
        slider.setPrefSize(400, 50);
        slider.valueProperty().addListener((ov, t, t1) -> {
            for (int i = 0; i < games.length; i++) {
                games[i].setGameSpeed(t1.doubleValue());
                rate.setText(Double.toString(t1.doubleValue()));
            }
        });
        VBox vbox = new VBox(rate, slider);
        return vbox;
    }
}
