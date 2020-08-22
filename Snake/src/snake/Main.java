package snake;

import ai.*;
import javafx.application.Application;
import static javafx.application.Application.launch;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.control.Slider;
import javafx.stage.Stage;

public class Main extends Application {

    public final Group ROOT = new Group(gameSpeedSlider());
    private static final boolean whileloop = false;
    Snake game = new Snake(6, 6, 32, whileloop);

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage stage) throws Exception {
        if (!whileloop) {
            Scene scene = new Scene(ROOT, 400, 50);
            stage.setScene(scene);
            stage.setResizable(false);
            stage.sizeToScene();
            stage.show();
            game.setAlwaysOnTop(true);
            game.setAlwaysOnTop(false);
        }
//        Naive naive = new Naive(game);
//        Genetic genetic = new Genetic(game);
//        REINFORCE pg = new REINFORCE(game);
        if (whileloop) {
            for (int i = 0; i < 3; i++) {
                Snake game = new Snake(6, 6, 32, whileloop);
                ActorCritic ac = new ActorCritic(game);
                Thread t = new Thread(() -> game.play());
                t.setDaemon(true);
                t.start();
            }
        }
        ActorCritic ac = new ActorCritic(game);
        game.play();
    }

    private Slider gameSpeedSlider() {
        Slider slider = new Slider();
        slider.setMin(1);
        slider.setMax(500);
        slider.setBlockIncrement(5);
        slider.setPrefSize(400, 50);
        slider.valueProperty().addListener((ov, t, t1) -> {
            game.setGameSpeed(t1.doubleValue());
        });
        return slider;
    }
}
