package ai;

import ai.NNlib.*;
import snake.Snake;

public class Genetic {

    private NN base = new NN("genetic", 0, .00001f, null, null,
            new Layer.Conv(1, 6, 6, 20, 3, 3, 1, 0, 0, Activations.TANH),//1-6-6 conv(s=1) 20-1-3-3 = 20-4-4
            new Layer.Flatten(20, 4, 4),
            new Layer.Dense(320, 100, Activations.TANH, Initializers.XAVIER),
            new Layer.Dense(100, 4, Activations.SIGMOID, Initializers.XAVIER)
    );
    private Snake game;
    private int gamesbeforesave = 1000;
    private float avg;
    private int populationsize = 1000;
    private NN[] population = new NN[populationsize];
    private float[] fitnesses = new float[populationsize];
    private int accumulatedlength = 0;
    private float mutationrate = .3f;
    private NN current;
    private int index = 0;

    public Genetic(Snake game) {
        base.loadInsideJar();
        this.game = game;
        game.setUpdate(() -> update());
        population[0] = base;
        for (int i = 1; i < populationsize; i++) {
            NN copy = base.clone();
            copy.mutate(2, mutationrate);
            population[i] = base.clone();
        }
        current = population[index];
    }

    public void update() {
        fitnesses[index] += getReward();
        if (game.isSnakeAlive()) {
            int action;
            if (Math.random() < .01) {
                action = (int) (Math.random() * 4);
            } else {
                float[][] actionscores = (float[][]) current.feedforward(new float[][][]{game.getGameState()});
                action = NNlib.argmax(actionscores);
            }
            game.setInput(action);
        } else {
            accumulatedlength += game.getLength();
            if (game.getGamesCount() != 0 && game.getGamesCount() % gamesbeforesave == 0) {
                System.out.println("Highest Length: " + game.getHighestLength());
                game.setHighScoreReset();
                avg = ((float) accumulatedlength) / gamesbeforesave;
                System.out.println("Average Length: " + avg);
                accumulatedlength = 0;
                base.saveInsideJar();
            }
            if (index == populationsize - 1) {
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

    public void reset() {
        index = (index + 1) % populationsize;
        current = population[index];
    }

    public void train() {
        base.copyParameters(population[NNlib.argmax(new float[][]{fitnesses})]);
        population[0] = base;
        for (int i = 1; i < populationsize; i++) {
            NN copy = base.clone();
            copy.mutate(2, mutationrate);
            population[i] = base.clone();
        }
        current = population[index];
    }
}
