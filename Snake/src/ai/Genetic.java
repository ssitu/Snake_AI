package ai;

import ai.NNlib.*;
import snake.Snake;

public class Genetic {

    private NN base = new NN("genetic", 0, 0, null, null,
            new Layer.Conv(1, 6, 6, 10, 3, 3, 1, 0, 0, Activations.TANH),//1-6-6 conv(s=1) 20-1-3-3 = 20-4-4
            new Layer.Flatten(),
            new Layer.Dense(4, Activations.SIGMOID, Initializers.XAVIER)
    );
    private Snake game;
    private float avg;
    private int populationsize = 10000;
    private NN[] population = new NN[populationsize];
    private float[] fitnesses = new float[populationsize];
    private int accumulatedlength = 0;
    private float mutationrate = .1f;
    private NN current;
    private int index = 0;

    public Genetic(Snake game) {
        base.loadInsideJar();
        this.game = game;
        game.setUpdate(() -> update());
        population[0] = base;
        for (int i = 1; i < populationsize; i++) {
            population[i] = base.clone();
            population[i].mutate(2, mutationrate);
        }
        current = population[index];
    }

    public void update() {
        fitnesses[index] += getReward();
        if (game.isSnakeAlive()) {
            int action;
            float[][] actionscores = (float[][]) current.feedforward(new float[][][]{game.getGameState()});
            action = NNlib.argmax(actionscores);
            game.setInput(action);
        } else {
            accumulatedlength += game.getLength();
            if (game.getGamesPlayed() != 0 && game.getGamesPlayed() % populationsize == 0) {
                System.out.println("Highest Length of Generation: " + game.getHighestLength());
                game.setHighScoreReset();
                avg = ((float) accumulatedlength) / populationsize;
                System.out.println("Average Length of Generation: " + avg);
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
                return -1f / game.HEIGHT / game.WIDTH;
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
        int index = NNlib.argmax(new float[][]{fitnesses});
        System.out.println("Highest fitness: " + fitnesses[index]);
        base.copyParameters(population[index]);
        population[0] = base;
        for (int i = 1; i < populationsize; i++) {
            population[i] = base.clone();
            population[i].mutate(2, mutationrate);
        }
        current = population[index];
        fitnesses = new float[populationsize];
    }
}
