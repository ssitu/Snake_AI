package ai;

import ai.NNlib.*;
import static ai.NNlib.*;
import java.util.ArrayList;
import snake.Snake;

public class REINFORCE {

    private NN nn = new NN("pg", 0, .00001f, LossFunctions.CROSSENTROPY(1), Optimizers.ADAM,
            new Layer.Conv(1, 6, 6, 50, 3, 3, 1, 0, 0, Activations.TANH),
            new Layer.Flatten(),
            new Layer.Dense(200, Activations.TANH, Initializers.XAVIER),
            new Layer.Dense(4, Activations.SOFTMAX, Initializers.XAVIER)
    );
    private final Snake GAME;
    private final ArrayList<float[][][]> STATES = new ArrayList<>();
    private final ArrayList<Integer> ACTIONS = new ArrayList<>();
    private final float DISCOUNT = .99f;
    private float totalreward = 0;
    private final int GAMESBEFORESAVE = 1000;
    private float avg = 0;
    private float accumulatedLength = 0;
    private float avgreward = 0;
    private final int PUNISHMENTADJUSTMENT = 0;
    private final boolean TRAIN = true;
    private final float AREA;
    private float ewma = 0;
    private final float EWMACOEFFICIENT = .0001f;
    private float punishment = 1;
    private int highestlength = 0;

    public REINFORCE(Snake game) {
        nn.loadInsideJar();
//        NNlib.showInfo(infoLayers, nn);
        this.GAME = game;
        game.setUpdate(() -> update());
        AREA = game.WIDTH * game.HEIGHT;
    }

    private void update() {
        totalreward += getReward();
        if (GAME.isSnakeAlive()) {
            float[][][] state = {copy(GAME.getGameState())};
            float[][] probabilities = (float[][]) nn.feedforward(state);
//            print(probabilities);
            STATES.add(state);
            int action = sampleProbabilities(probabilities[0], nn.getRandom());
            GAME.setInput(action);
            ACTIONS.add(action);
        } else {
            accumulatedLength += GAME.getLength();
            avgreward += totalreward;
            if (GAME.getGamesCount() != 0 && GAME.getGamesCount() % GAMESBEFORESAVE == 0) {
                highestlength = Math.max(highestlength, GAME.getHighestLength());
                System.out.println("Highest Length: " + highestlength);
                System.out.println("Batch Highest Length: " + GAME.getHighestLength());
                GAME.setHighScoreReset();
                avg = accumulatedLength / GAMESBEFORESAVE;
                System.out.println("Batch Average Length: " + avg);
                accumulatedLength = 0;
                System.out.println("EW Moving Average: " + ewma);
                avgreward /= GAMESBEFORESAVE;
                System.out.println("Batch Average Reward: " + avgreward);
                System.out.println("");
                if (avgreward > 1 && PUNISHMENTADJUSTMENT == 3) {
                    punishment += .5;
                }
                avgreward = 0;
            }
            if (TRAIN) {
                train();
            }
            reset();
        }
    }

    private float getReward() {
        if (GAME.isSnakeAlive()) {
            if (GAME.isAppleEaten()) {
                return 1;
            } else {
                return -1 / AREA;
            }
        } else {
            if (GAME.isGameWon()) {
                System.out.println("Game Won!");
                return 10;
            } else {
                ewma = EWMACOEFFICIENT * GAME.getLength() + (1 - EWMACOEFFICIENT) * ewma;
                if (PUNISHMENTADJUSTMENT == 1) {
                    if (avg == 0) {
                        return -totalreward;
                    } else {
                        return -avg / 2;
                    }
                } else if (PUNISHMENTADJUSTMENT == 2) {
                    return -ewma;
                } else if (PUNISHMENTADJUSTMENT == 3) {
                    return -punishment;
                } else {
                    return -2;
                }
            }
        }
    }

    private void train() {
        int timesteps = STATES.size() - 1;
        float reward = totalreward;
        for (int t = timesteps; t >= 0; t--) {
            float[][] labels = new float[1][4];
            try {
                labels[0][ACTIONS.get(t)] = reward;
                nn.backpropagation(STATES.get(t), labels);
            } catch (Exception e) {
                print((float[][]) nn.feedforward(STATES.get(t)));
            }
            reward *= DISCOUNT;
//            if (Math.abs(reward) < 0.0001) {
//                break;
//            }
        }
        if (GAME.getGamesCount() % GAMESBEFORESAVE == 0) {
            nn.saveInsideJar();
        }
    }

    private void reset() {
        STATES.clear();
        ACTIONS.clear();
        totalreward = 0;
    }
}
