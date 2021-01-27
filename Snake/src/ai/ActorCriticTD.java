package ai;

import ai.NNlib.*;
import static ai.NNlib.*;
import java.io.FileWriter;
import java.io.IOException;
import snake.Snake;

public class ActorCriticTD {

    private static NN pn;
    private static NN vn;
    private final Snake GAME;
    private final float DISCOUNT = .99f;
    private float totalreward = 0;
    private final int GAMESBEFORESAVE = 500;
    private final boolean TRAIN = true;
    private final float AREA;
    private boolean firstTimeStep = true;
    //statistics related
    private boolean firstGame = true;
    private FileWriter fw;
    private float accumulatedLength = 0;
    private float avgLength = 0;
    private float avgReward = 0;
    private float ewma = 0;
    private final float EWMACOEFFICIENT = .0001f;
    private int highestLength = 0;
    private float highestAdvantage = Float.NEGATIVE_INFINITY;
    private float highestReward = Float.NEGATIVE_INFINITY;
    private float highestValue = Float.NEGATIVE_INFINITY;
    //states, actions, and rewards
    private float[][][] s;
    private int a;
    private float r_;
    private float[][][] s_;

    public ActorCriticTD(Snake game) {
        this.GAME = game;
        try {
            fw = new FileWriter("ActorCriticTD.csv", true);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
        try {
            GAME.setGamesPlayed(Integer.parseInt(Helper.lastLine("ActorCriticTD.csv").split(",")[0]));
        } catch (IOException e) {
            e.printStackTrace();
        }
//        NNlib.showInfo(infoLayers, nn);
        game.setUpdate(() -> update());
        AREA = game.WIDTH * game.HEIGHT;
        createActor(game.WIDTH, game.HEIGHT);
        pn.loadInsideJar();
        createCritic(game.WIDTH, game.HEIGHT);
        vn.loadInsideJar();
    }

    private void createActor(int gameWidth, int gameHeight) {
        pn = new NN("ac_actor", 0, .00001f, LossFunctions.CROSSENTROPY(1), Optimizers.ADAM,
                new Layer.Conv(1, gameWidth, gameHeight, 5, 3, 3, 1, 1, 1, Activations.TANH),
                new Layer.Flatten(),
                new Layer.Dense(100, Activations.TANH, Initializers.XAVIER),
                new Layer.Dense(4, Activations.SOFTMAX, Initializers.XAVIER)
        );
    }

    private void createCritic(int gameWidth, int gameHeight) {
        vn = new NN("ac_critic", 0, .00001f, LossFunctions.HUBERPSEUDO(.5), Optimizers.ADAM,
                new Layer.Flatten(1, gameWidth, gameHeight),
                new Layer.Dense(20, Activations.TANH, Initializers.XAVIER),
                new Layer.Dense(20, Activations.TANH, Initializers.XAVIER),
                new Layer.Dense(1, Activations.LINEAR, Initializers.HE)
        );
    }

    private void update() {
        if (firstTimeStep && GAME.isSnakeAlive() && !GAME.isGameWon()) {
            s = new float[][][]{copy(GAME.getGameState())};
            float[][] probabilities = (float[][]) pn.feedforward(s);
//            print(probabilities);
            a = sampleProbabilities(probabilities[0], pn.getRandom());
            GAME.setInput(a);
            firstTimeStep = false;
        } else {
            r_ = getReward();
            totalreward += r_;
            s_ = new float[][][]{copy(GAME.getGameState())};
            if (TRAIN) {
                train(s, a, r_, s_);
            }
            s = s_;
            float[][] probabilities = (float[][]) pn.feedforward(s);
            a = sampleProbabilities(probabilities[0], pn.getRandom());
            GAME.setInput(a);
        }
        if (!GAME.isSnakeAlive() || GAME.isGameWon()) {
            afterGame();
            firstGame = false;
        }
    }

    private void afterGame() {
        accumulatedLength += GAME.getLength();
        avgReward += totalreward;
        highestReward = Math.max(highestReward, totalreward);
        ewma = EWMACOEFFICIENT * GAME.getLength() + (1 - EWMACOEFFICIENT) * ewma;
        if (!firstGame && GAME.getGamesPlayed() % GAMESBEFORESAVE == 0) {
            highestLength = Math.max(highestLength, GAME.getHighestLength());
            avgReward /= GAMESBEFORESAVE;
            avgLength = accumulatedLength / GAMESBEFORESAVE;
            System.out.println("Highest Length: " + highestLength);
            System.out.println("Highest Reward: " + highestReward);
            System.out.println("EW Moving Average Length: " + ewma);
            System.out.println("Batch Highest Length: " + GAME.getHighestLength());
            System.out.println("Batch Average Length: " + avgLength);
            System.out.println("Batch Average Reward: " + avgReward);
            System.out.println("Batch Highest State-Value: " + highestValue);
            System.out.println("Batch Highest Advantage: " + highestAdvantage);
            System.out.println("");
            Helper.writeStatsToCSV(fw, GAME, avgLength);
            accumulatedLength = 0;
            avgReward = 0;
            GAME.setHighScoreReset();
            highestValue = Float.NEGATIVE_INFINITY;
            highestAdvantage = Float.NEGATIVE_INFINITY;
        }
        save();
        reset();
    }

    private float getReward() {
        if (GAME.isSnakeAlive()) {//alive
            if (GAME.isAppleEaten()) {
                return 1;
            } else {
//                return -1 / AREA;
                return 0;
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

    private void train(float[][][] s, int a, float r_, float[][][] s_) {
        float[][] labels = new float[1][4];
        float v = ((float[][]) vn.feedforward(s))[0][0];
        float v_ = ((float[][]) vn.feedforward(s_))[0][0];
        float advantage = r_ + DISCOUNT * v_ - v;
        try {
            labels[0][a] = advantage;
            pn.backpropagation(s, labels);
        } catch (Exception e) {
            print((float[][]) pn.feedforward(s));
        }
        vn.backpropagation(s, new float[][]{{r_ + DISCOUNT * v_}});
        //saving statistics            
        highestValue = Math.max(highestValue, v);
        highestAdvantage = Math.max(highestAdvantage, advantage);
    }

    private void save() {
        if (GAME.getGamesPlayed() % GAMESBEFORESAVE == 0) {
            pn.saveInsideJar();
            vn.saveInsideJar();
        }
    }

    private void reset() {
        totalreward = 0;
        firstTimeStep = true;
    }
}
