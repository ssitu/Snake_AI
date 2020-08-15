package ai;

import ai.NNlib.*;
import static ai.NNlib.*;
import java.util.ArrayList;
import snake.Snake;

public class PolicyGradient {

    private NN nn = new NN("pg", 0, .00001f, LossFunction.CROSSENTROPY(1), Optimizer.VANILLA,
            new Layer.Conv(1, 6, 6, 20, 3, 3, 1, 0, 0, Activation.TANH),//1-6-6 conv(s=1) 20-1-3-3 = 20-4-4
            new Layer.Flatten(),
            new Layer.Dense(101, Activation.TANH, Initializer.XAVIER),
            new Layer.Dense(4, Activation.SOFTMAX, Initializer.XAVIER)
    );
    private Snake game;
    private ArrayList<float[][][]> states = new ArrayList<>();
    private ArrayList<Integer> actions = new ArrayList<>();
    private float discount = .99f;
    private float totalreward = 0;
    private final int GAMESBEFORESAVE = 1000;
    private float avg = 0;
    private float accumulatedLength = 0;
    private float avgreward = 0;
    private int punishmentadjustment = 0;
    private boolean train = true;
    private float area;
    private float ewma = 0;
    private final float ewmacoefficient = .05f;

    public PolicyGradient(Snake game) {
        nn.loadInsideJar();
//        NNlib.showInfo(infoLayers, nn);
        this.game = game;
        game.setUpdate(() -> update());
        area = game.WIDTH * game.HEIGHT;
    }

    private void update() {
        totalreward += getReward();
        if (game.isSnakeAlive()) {
            float[][][] state = {copy2d(game.getGameState())};
            float[][] probabilities = (float[][]) nn.feedforward(state);
//            print(probabilities);
            states.add(state);
            int action = sampleProbabilities(probabilities[0], nn.getRandom());
            game.setInput(action);
            actions.add(action);
        } else {
            accumulatedLength += game.getLength();
            avgreward += totalreward;
            if (game.getGamesCount() != 0 && game.getGamesCount() % GAMESBEFORESAVE == 0) {
                System.out.println("Highest Length: " + game.getHighestLength());
                game.setHighScoreReset();
                avg = accumulatedLength / GAMESBEFORESAVE;
                System.out.println("Average Length: " + avg);
                accumulatedLength = 0;
                System.out.println("EW Moving Average: " + ewma);
                avgreward /= GAMESBEFORESAVE;
                System.out.println("Average Reward: " + avgreward);
                avgreward = 0;
            }
            if (train) {
                train();
            }
            reset();
        }
    }

    private float getReward() {
        if (game.isSnakeAlive()) {
            if (game.isAppleEaten()) {
                return 1;
            } else {
                return -1 / area;
            }
        } else {
            if (game.isGameWon()) {
                System.out.println("WOOHOO");
                return 10;
            } else {
                ewma = ewmacoefficient * game.getLength() + (1 - ewmacoefficient) * ewma;
                if (punishmentadjustment == 1) {
                    if (avg == 0) {
                        return -totalreward;
                    } else {
                        return -avg + 5;
                    }
                } else if (punishmentadjustment == 2) {
                    return -ewma;
                } else {
                    return -10;
                }
            }
        }
    }

    private void train() {
        int timesteps = states.size() - 1;
        float reward = totalreward;
        for (int t = timesteps; t >= 0; t--) {
            float[][] labels = new float[1][4];
            try {
                labels[0][actions.get(t)] = reward;
                nn.backpropagation(states.get(t), labels);
            } catch (Exception e) {
                print((float[][]) nn.feedforward(states.get(t)));
            }
            reward *= discount;
//            if (Math.abs(reward) < 0.0001) {
//                break;
//            }
        }
        if (game.getGamesCount() % GAMESBEFORESAVE == 0) {
            nn.saveInsideJar();
        }
    }

    private void reset() {
        states.clear();
        actions.clear();
        totalreward = 0;
    }
}
