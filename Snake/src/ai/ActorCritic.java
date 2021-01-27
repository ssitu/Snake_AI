package ai;

import ai.NNlib.*;
import static ai.NNlib.*;
import java.util.ArrayList;
import snake.Snake;

public class ActorCritic {

    private static NN pn;
    private static NN vn;
    private final Snake GAME;
    private final ArrayList<float[][][]> STATES = new ArrayList<>();
    private final ArrayList<Integer> ACTIONS = new ArrayList<>();
    private final ArrayList<Float> REWARDS = new ArrayList<>();
    private final float DISCOUNT = .99f;
    private float totalreward = 0;
    private final int GAMESBEFORESAVE = 500;
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
        pn = createActor(game.WIDTH, game.HEIGHT);
        pn.loadInsideJar();
        vn = createCritic(game.WIDTH, game.HEIGHT);
        vn.loadInsideJar();
    }

    private NN createActor(int gameWidth, int gameHeight) {
        return new NN("ac_actor", 0, .00005f, LossFunctions.CROSSENTROPY(1), Optimizers.ADAM,
                new Layer.Conv(1, gameWidth, gameHeight, 10, 3, 3, 1, 1, 1, Activations.TANH),
                new Layer.Flatten(),
                new Layer.Dense(200, Activations.TANH, Initializers.XAVIER),
                new Layer.Dense(4, Activations.SOFTMAX, Initializers.XAVIER)
        );
    }

    private NN createCritic(int gameWidth, int gameHeight) {
        return new NN("ac_critic", 0, .00005f, LossFunctions.QUADRATIC(.5), Optimizers.ADAM,
                new Layer.Flatten(1, gameWidth, gameHeight),
                new Layer.Dense(100, Activations.TANH, Initializers.XAVIER),
                new Layer.Dense(100, Activations.TANH, Initializers.XAVIER),
                new Layer.Dense(1, Activations.LINEAR, Initializers.VANILLA)
        );
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
            if (GAME.getGamesPlayed() != 0 && GAME.getGamesPlayed() % GAMESBEFORESAVE == 0) {
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
        if (GAME.isSnakeAlive()) {//alive
            if (GAME.isAppleEaten()) {
                return 1;
            } else {
                return -1 / AREA;
            }
        } else {
            if (GAME.isGameWon()) {//win
                System.out.println("Game Won!");
                return 10;
            } else {//lose
                return -1;
            }
        }
    }

    private void train() {
        int T = STATES.size() - 1;
        for (int t = 0; t < T; t++) {
            float[][] labels = new float[1][4];
            float[][][] s_ = STATES.get(t + 1);
            float[][][] s = STATES.get(t);
            float v_ = ((float[][]) vn.feedforward(s_))[0][0];
            float v = ((float[][]) vn.feedforward(s))[0][0];
            float r_ = REWARDS.get(t + 1);
            float advantage = r_ + DISCOUNT * v_ - v;
            try {
                labels[0][ACTIONS.get(t)] = advantage;
                pn.backpropagation(s, labels);
            } catch (Exception e) {
                print((float[][]) pn.feedforward(s));
            }
            vn.backpropagation(s, new float[][]{{r_ + DISCOUNT * v_}});
            //saving statistics            
            highestvalue = Math.max(highestvalue, v);
            highestadvantage = Math.max(highestadvantage, advantage);
        }
        if (GAME.getGamesPlayed() % GAMESBEFORESAVE == 0) {
            pn.saveInsideJar();
            vn.saveInsideJar();
        }
    }
//    private void train() {
//        int timesteps = STATES.size() - 1;
//        float reward = totalreward;
//        for (int t = timesteps; t >= 0; t--) {
//            float[][] labels = new float[1][4];
//            float[][][] state = STATES.get(t);
//            float v = ((float[][]) vn.feedforward(state))[0][0];
//            highestvalue = Math.max(highestvalue, v);
//            float advantage = reward - v;
//            highestadvantage = Math.max(highestadvantage, advantage);
//            try {
//                labels[0][ACTIONS.get(t)] = advantage;
//                pn.backpropagation(state, labels);
//            } catch (Exception e) {
//                print((float[][]) pn.feedforward(state));
//            }
//            reward *= DISCOUNT;
//            vn.backpropagation(state, new float[][]{{REWARDS.get(t + 1) + DISCOUNT * v}});
//        }
//        if (GAME.getGamesCount() % GAMESBEFORESAVE == 0) {
//            pn.saveInsideJar();
//            vn.saveInsideJar();
//        }
//    }

    private void reset() {
        STATES.clear();
        ACTIONS.clear();
        totalreward = 0;
    }
}
