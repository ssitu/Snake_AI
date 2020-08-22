package ai;

import snake.Snake;

public class Naive {

    private Snake game;
    private int[][] actionmap;
    private int width;
    private int height;
    private boolean firstFrame = true;

    public Naive(Snake game) {
        this.game = game;
        game.setUpdate(() -> update());
        width = game.WIDTH;
        height = game.HEIGHT;
        if (width % 2 != 0 && height % 2 != 0) {
            throw new IllegalArgumentException("No cycle can be found");
        }
        actionmap = new int[height][width];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int action;
                if (i != 0 && j == 0) {
                    action = 0;
                } else if (i != height - 1 && ((i % 2 == 0 && j == width - 1) || (i % 2 != 0 && j == 1))) {
                    action = 1;
                } else if ((i % 2 != 0 && j > 1) || i == height - 1 && j == 1) {
                    action = 2;
                } else {
                    action = 3;
                }
                actionmap[i][j] = action;
                System.out.print(action + " ");
            }
            System.out.println("");
        }
    }

    public void update() {
        if (game.isSnakeAlive() || !game.isGameWon()) {
            if (firstFrame) {
                game.setInput(0);
                firstFrame = false;
            } else {
                int[] headxy = game.getHeadLoc();
                game.setInput(actionmap[headxy[1]][headxy[0]]);
            }
        } else {
            firstFrame = true;
        }
    }
}
