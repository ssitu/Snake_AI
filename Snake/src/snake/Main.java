package snake;

import ai.PolicyGradient;
import javafx.application.Application;
import static javafx.application.Application.launch;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.control.Slider;
import javafx.stage.Stage;

public class Main extends Application {
    
    public final Group ROOT = new Group(gameSpeedSlider());
    private Snake game = new Snake(6, 6, 100);
    
    public static void main(String[] args) {
        launch(args);
    }
    
    @Override
    public void start(Stage stage) throws Exception {
        PolicyGradient pg = new PolicyGradient(game);
        Scene scene = new Scene(ROOT, 400, 50);
        stage.setScene(scene);
        stage.setResizable(false);
        stage.sizeToScene();
        stage.show();
        game.play();
        game.setAlwaysOnTop(true);
        game.setAlwaysOnTop(false);
    }
    
    private Slider gameSpeedSlider() {
        Slider slider = new Slider();
        slider.setMin(1);
        slider.setMax(500);
        slider.setBlockIncrement(2);
        slider.setPrefSize(400, 50);
        slider.valueProperty().addListener((ov, t, t1) -> {
            game.setGameSpeed(t1.doubleValue());
        });
        return slider;
    }
}
