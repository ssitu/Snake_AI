package ai;

import ai.NNlib.*;
import static ai.NNlib.*;
import java.io.FileWriter;
import java.io.IOException;
import snake.Snake;

public class PPO {

    private static NN pnOld;
    private static NN pn;
    private static NN vn;
    private final Snake GAME;
    private final float DISCOUNT = .99f;
    private final int GAMESBEFORESAVE = 500;
    private final boolean TRAIN = false;
    private final float AREA;
    private boolean firstTimeStep = true;
    //statistics related
    private float totalReward = 0;
    private float accumulatedLength = 0;
    private float ewmaLength = 0;
    private final float EWMACOEFFICIENT = .0001f;
    private int highestLength = 0;
    private float highestAdvantage = Float.NEGATIVE_INFINITY;
    private float highestReward = Float.NEGATIVE_INFINITY;
    private float highestStateValue = Float.NEGATIVE_INFINITY;
    private float avgLength = 0;
    private float avgReward = 0;
    private FileWriter fw;
    private final String csvPath = "PPO.csv";
    private boolean firstGame = true;
    //states, actions, and rewards
    private float[][][] s;//stack of the current frame and the previous frame to show movement
    private int a;
    private float r_;
    private float[][][] s_;
    //PPO
    private float[][] pnOldProbabilities;
    private final float EPSILON = .2f;
    private final int GAMESBEFORESYNC = 10;
    private final float ENTROPYWEIGHT = 0;

    private float PPOAlternativeClip(float advantage) {
        if (advantage >= 0) {
            return (1 + EPSILON) * advantage;
        } else {
            return (1 - EPSILON) * advantage;
        }
    }
    private final LossFunctions.LossFunction ppoLoss = (outputs, targets) -> {
        int action = 0;
        final int totalActions = 4;
        for (int i = 1; i < totalActions; i++) {//find the one value that is not a zero
            if (targets[0][i] != 0) {
                action = i;
            }
        }
        float[][] ppoGradient = new float[1][totalActions];
        //PPO
        float advantage = targets[0][action];
        float oldProbability = pnOldProbabilities[0][action];
        float ratio = outputs[0][action] / oldProbability;
        float ratioAdvantage = ratio * advantage;
        float limit = PPOAlternativeClip(advantage);
        boolean clipped = limit < ratioAdvantage;
        double ppoObjective;
        if (clipped) {
            ppoObjective = limit;
            ppoGradient[0][action] = 0;
        } else {
            ppoObjective = ratioAdvantage;
            ppoGradient[0][action] = advantage / oldProbability;
        }
        //Entropy bonus
        float[][] entropy = function(outputs, x -> -x * (float) Math.log(x));
        double entropyBonus = sum(entropy);
        float[][] entropyGradient = function(outputs, x -> (float) -Math.log(x) - 1);
        //Prepare objective and gradient
        double objective = ppoObjective + ENTROPYWEIGHT * entropyBonus;
        float[][] gradient = add(ppoGradient, scale(ENTROPYWEIGHT, entropyGradient));
        //add negative signs for minimizing
        return new Object[]{-objective, scale(-1, gradient)};
    };

    public PPO(Snake game) {
        this.GAME = game;
        try {
            fw = new FileWriter(csvPath, true);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
        try {
            GAME.setGamesPlayed(Integer.parseInt(Helper.lastLine(csvPath).split(",")[0]));
        } catch (IOException e) {
            e.printStackTrace();
        }
//        NNlib.showInfo(infoLayers, nn);
        game.setUpdate(() -> update());
        AREA = game.WIDTH * game.HEIGHT;
        createActor(game.WIDTH, game.HEIGHT);
        pn.loadInsideJar();
        pnOld = pn.clone();
        createCritic(game.WIDTH, game.HEIGHT);
        vn.loadInsideJar();
    }

    private void createActor(int gameWidth, int gameHeight) {
        pn = new NN("ppo_actor", 0, .0001f, ppoLoss, Optimizers.ADAM,
                new Layer.Conv(2, gameWidth, gameHeight, 10, 2, 2, 1, 1, 1, Activations.TANH),
                new Layer.Flatten(),
                new Layer.Dense(200, Activations.TANH, Initializers.XAVIER),
                new Layer.Dense(4, Activations.SOFTMAX, Initializers.XAVIER)
        );
    }

    private void createCritic(int gameWidth, int gameHeight) {
        vn = new NN("ppo_critic", 0, .0001f, LossFunctions.HUBER(.5), Optimizers.ADAM,
                new Layer.Flatten(2, gameWidth, gameHeight),
                new Layer.Dense(100, Activations.TANH, Initializers.XAVIER),
                new Layer.Dense(100, Activations.TANH, Initializers.XAVIER),
                new Layer.Dense(1, Activations.LINEAR, Initializers.VANILLA)
        );
    }

    private void update() {
//        System.out.println(GAME.getLength());
        if (firstTimeStep && GAME.isSnakeAlive() && !GAME.isGameWon()) {
            s = new float[][][]{GAME.getGameState(), GAME.getGameState()};
            float[][] probabilities = (float[][]) pn.feedforward(s);
//            print(probabilities);
            a = sampleProbabilities(probabilities[0], pn.getRandom());
            GAME.setInput(a);
            firstTimeStep = false;
        } else {
            r_ = getReward();
            totalReward += r_;
            s_ = new float[][][]{s[1], GAME.getGameState()};
            if (TRAIN) {
                train(s, a, r_, s_);
            }
            s = s_;
            float[][] probabilities = (float[][]) pn.feedforward(s);
//            print(probabilities);
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
        avgReward += totalReward;
        highestReward = Math.max(highestReward, totalReward);
        ewmaLength = EWMACOEFFICIENT * GAME.getLength() + (1 - EWMACOEFFICIENT) * ewmaLength;
        if (!firstGame && GAME.getGamesPlayed() % GAMESBEFORESAVE == 0) {
            highestLength = Math.max(highestLength, GAME.getHighestLength());
            avgLength = accumulatedLength / GAMESBEFORESAVE;
            avgReward /= GAMESBEFORESAVE;
            System.out.println("Highest Length: " + highestLength);
            System.out.println("Highest Reward: " + highestReward);
            System.out.println("EW Moving Average Length: " + ewmaLength);
            System.out.println("Batch Highest Length: " + GAME.getHighestLength());
            System.out.println("Batch Average Length: " + avgLength);
            System.out.println("Batch Average Reward: " + avgReward);
            if (TRAIN) {
                System.out.println("Batch Highest State-Value: " + highestStateValue);
                System.out.println("Batch Highest Advantage: " + highestAdvantage);
            }
            System.out.println("");
            if (TRAIN) {
                Helper.writeStatsToCSV(fw, GAME, avgLength);
            }
            GAME.setHighScoreReset();
            accumulatedLength = 0;
            avgReward = 0;
            highestStateValue = Float.NEGATIVE_INFINITY;
            highestAdvantage = Float.NEGATIVE_INFINITY;
        }
        if (TRAIN) {
            save();
        }
        reset();
    }

    private float getReward() {
//        System.out.println(GAME.getLength());
        if (GAME.isSnakeAlive()) {//alive
            if (GAME.isGameWon()) {//win
                System.out.println("Game Won!");
                return 10;
            } else if (GAME.isAppleEaten()) {//apple
                return 1;
            } else {//moved to an empty space
                return -1 / AREA;
//                return 0;
            }
        } else {//lose
            return -1;
//                return 0;
        }
    }

    private void train(float[][][] s, int a, float r_, float[][][] s_) {
        float[][] labels = new float[1][4];
        float v = ((float[][]) vn.feedforward(s))[0][0];
        float v_ = ((float[][]) vn.feedforward(s_))[0][0];
        float advantage = r_ + DISCOUNT * v_ - v;
        pnOldProbabilities = (float[][]) pnOld.feedforward(s);
        try {
            labels[0][a] = advantage;
            pn.backpropagation(s, labels);
        } catch (Exception e) {
            print((float[][]) pn.feedforward(s));
            e.printStackTrace();
        }
        vn.backpropagation(s, new float[][]{{r_ + DISCOUNT * v_}});
        //saving statistics            
        highestStateValue = Math.max(highestStateValue, v);
        highestAdvantage = Math.max(highestAdvantage, advantage);
    }

    private void save() {
        if (GAME.getGamesPlayed() % GAMESBEFORESAVE == 0) {
            System.out.println("Saving...");
            pn.saveInsideJar();
            vn.saveInsideJar();
            System.out.println("Done Saving.");
        }
    }

    private void reset() {
        totalReward = 0;
        firstTimeStep = true;
        if (GAME.getGamesPlayed() % GAMESBEFORESYNC == 0) {
            pnOld.copyParameters(pn);
        }
    }
}
