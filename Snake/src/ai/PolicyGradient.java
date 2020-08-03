package ai;

import ai.NNlib.*;
import static ai.NNlib.*;
import java.util.ArrayList;
import snake.Snake;

public class PolicyGradient {

    private NN nn = new NN("pg", 0, .00001f, LossFunction.CROSSENTROPY(1), Optimizer.NESTEROV,
            new Layer.Conv(20, 1, 3, 3, 1, 0, 0, Activation.TANH),//1-6-6 conv(s=1) 20-1-3-3 = 20-4-4
            new Layer.Flatten(20, 4, 4),
            new Layer.Dense(320, 100, Activation.TANH, Initializer.XAVIER),
            new Layer.Dense(100, 4, Activation.SOFTMAX, Initializer.XAVIER)
    );
//    private NN nn = new NN("pg", 0, .00001f, LossFunction.CROSSENTROPY(1), Optimizer.AMSGRAD,
//            new Layer.Conv(20, 1, 3, 3, 1, 0, 0, Activation.TANH),//1-6-6 conv(s=1) 20-1-3-3 = 20-4-4
//            new Layer.Flatten(20, 4, 4),
//            new Layer.Dense(320, 4, Activation.SOFTMAX, Initializer.XAVIER)
//    );
    private Snake game;
    private ArrayList<float[][][]> states = new ArrayList<>();
    private ArrayList<Integer> actions = new ArrayList<>();
    private int iterations = 0;
    private float discount = (float) .99;
    private float totalreward = 0;

    public PolicyGradient(Snake game) {
        nn.loadInsideJar();
//        NNlib.showInfo(infoLayers, nn);
        this.game = game;
        game.setUpdate(() -> update());
    }

    private void update() {
        totalreward += getReward();
        if (game.isSnakeAlive()) {
            float[][][] state = {copy2d(game.getGameState())};
            float[][] probabilities = (float[][]) nn.feedforward(state);
//            print(probabilities);
            states.add(state);
            int action = sampleProbabilities(probabilities[0], nn.getRandom());
            game.inputKey(action);
            actions.add(action);
        } else {
            if (game.getGamesCount() != 0 && game.getGamesCount() % 1000 == 0) {
                System.out.println("Highest Length: " + game.getHighestLength());
            }
            train();
            reset();
        }
    }

    private float getReward() {
        if (game.isSnakeAlive()) {
            if (game.isAppleEaten()) {
                return 1;
            } else {
                return 0;
            }
        } else {
            if (game.isGameWon()) {
                System.out.println("WOOHOO");
                return 10;
            } else {
                return -1;
            }
        }
    }

    private void train() {
        int timesteps = states.size() - 1;
        float reward = totalreward;
        for (int t = timesteps; t >= 0; t--) {
            float[][] labels = new float[1][4];
            labels[0][actions.get(t)] = reward;
            nn.backpropagation(states.get(t), labels);
            reward *= discount;
            if (Math.abs(reward) < 0.01) {
                break;
            }
        }
        iterations++;
        if (iterations % 100 == 0) {
            nn.saveInsideJar();
        }
    }

    private void reset() {
        states.clear();
        actions.clear();
        totalreward = 0;
    }
}
