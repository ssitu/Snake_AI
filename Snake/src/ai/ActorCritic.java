package ai;

import ai.NNlib.*;
import static ai.NNlib.*;
import java.util.ArrayList;
import snake.Snake;

public class ActorCritic {

    private static NN pn = new NN("ac_actor", 0, .000001f, LossFunctions.CROSSENTROPY(1), Optimizers.ADAM,
            new Layer.Conv(1, 6, 6, 10, 3, 3, 1, 0, 0, Activations.TANH),
            new Layer.Flatten(),
            new Layer.Dense(200, Activations.TANH, Initializers.XAVIER),
            new Layer.Dense(4, Activations.SOFTMAX, Initializers.XAVIER)
    );
    private static NN vn = new NN("ac_critic", 0, .0001f, LossFunctions.QUADRATIC(.5), Optimizers.ADAM,
            new Layer.Flatten(1, 6, 6),
            new Layer.Dense(100, Activations.TANH, Initializers.XAVIER),
            new Layer.Dense(100, Activations.TANH, Initializers.XAVIER),
            new Layer.Dense(1, Activations.LINEAR, Initializers.VANILLA)
    );

    static {
        pn.loadInsideJar();
        vn.loadInsideJar();
    }
    private final Snake GAME;
    private final ArrayList<float[][][]> STATES = new ArrayList<>();
    private final ArrayList<Integer> ACTIONS = new ArrayList<>();
    private final ArrayList<Float> REWARDS = new ArrayList<>();
    private final float DISCOUNT = .99f;
    private float totalreward = 0;
    private final int GAMESBEFORESAVE = 1000;
    private float avg = 0;
    private float accumulatedLength = 0;
    private float avgreward = 0;
    private final boolean TRAIN = true;
    private final float AREA;
    private float ewma = 0;
    private final float EWMACOEFFICIENT = .0001f;
    private int highestlength = 0;
    private float highestadvantage = Float.NEGATIVE_INFINITY;
    private float highestreward = Float.NEGATIVE_INFINITY;
    private float highestvalue = Float.NEGATIVE_INFINITY;

    public ActorCritic(Snake game) {
//        NNlib.showInfo(infoLayers, nn);
        this.GAME = game;
        game.setUpdate(() -> update());
        AREA = game.WIDTH * game.HEIGHT;
    }

    private void update() {
        float reward = getReward();
        REWARDS.add(reward);
        totalreward += reward;
        if (GAME.isSnakeAlive() && !GAME.isGameWon()) {
            float[][][] state = {copy(GAME.getGameState())};
            float[][] probabilities = (float[][]) pn.feedforward(state);
//            print(probabilities);
            STATES.add(state);
            int action = sampleProbabilities(probabilities[0], pn.getRandom());
            GAME.setInput(action);
            ACTIONS.add(action);
        } else {
            accumulatedLength += GAME.getLength();
            avgreward += totalreward;
            highestreward = Math.max(highestreward, totalreward);
            ewma = EWMACOEFFICIENT * GAME.getLength() + (1 - EWMACOEFFICIENT) * ewma;
            if (GAME.getGamesCount() != 0 && GAME.getGamesCount() % GAMESBEFORESAVE == 0) {
                highestlength = Math.max(highestlength, GAME.getHighestLength());
                System.out.println("Highest Length: " + highestlength);
                System.out.println("Highest Reward: " + highestreward);
                System.out.println("EW Moving Average Length: " + ewma);
                System.out.println("Batch Highest Length: " + GAME.getHighestLength());
                GAME.setHighScoreReset();
                avg = accumulatedLength / GAMESBEFORESAVE;
                System.out.println("Batch Average Length: " + avg);
                accumulatedLength = 0;
                avgreward /= GAMESBEFORESAVE;
                System.out.println("Batch Average Reward: " + avgreward);
                avgreward = 0;
                System.out.println("Batch Highest State-Value: " + highestvalue);
                highestvalue = Float.NEGATIVE_INFINITY;
                System.out.println("Batch Highest Advantage: " + highestadvantage);
                highestadvantage = Float.NEGATIVE_INFINITY;
                System.out.println("");
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
                return -1;
            }
        }
    }

    private void train() {
        int timesteps = STATES.size() - 1;
        float reward = totalreward;
        float v = 0;
        for (int t = timesteps; t >= 0; t--) {
            float[][] labels = new float[1][4];
            float[][][] state = STATES.get(t);
//            System.out.println(REWARDS.get(t + 1) + " + " + DISCOUNT + " * " + v + " = " + (REWARDS.get(t + 1) + DISCOUNT * v));
//            System.out.println("before " + ((float[][]) vn.feedforward(state))[0][0]);
            vn.backpropagation(state, new float[][]{{REWARDS.get(t + 1) + DISCOUNT * v}});
//            System.out.println("after " + ((float[][]) vn.feedforward(state))[0][0]);
            v = ((float[][]) vn.feedforward(state))[0][0];
            highestvalue = Math.max(highestvalue, v);
            float advantage = reward - v;
            highestadvantage = Math.max(highestadvantage, advantage);
            try {
                labels[0][ACTIONS.get(t)] = advantage;
                pn.backpropagation(state, labels);
            } catch (Exception e) {
                print((float[][]) pn.feedforward(state));
            }
            reward *= DISCOUNT;
//            if (Math.abs(reward) < 0.0001) {
//                break;
//            }
        }
        if (GAME.getGamesCount() % GAMESBEFORESAVE == 0) {
            pn.saveInsideJar();
            vn.saveInsideJar();
        }
    }

    private void reset() {
        STATES.clear();
        ACTIONS.clear();
        totalreward = 0;
    }
}
