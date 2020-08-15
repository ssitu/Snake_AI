package snake;

import java.util.ArrayList;
import java.util.HashMap;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.scene.Group;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.input.KeyCode;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.stage.Stage;
import javafx.util.Duration;

public class Snake extends Stage {

    private final Grid grid;
    private Timeline loop = new Timeline(new KeyFrame(Duration.millis(200), (t) -> update()));
    private HashMap<KeyCode, Boolean> keys = new HashMap();
    private final int[] STARTDIRECTION = {-1, 0};
    private int[] snakedirection = vectorOpposite(STARTDIRECTION);
    private int event = 0;
    private KeyCode up = KeyCode.W;
    private KeyCode down = KeyCode.S;
    private KeyCode left = KeyCode.A;
    private KeyCode right = KeyCode.D;
    private Runnable externalupdate = () -> {
    };
    private int completedgames = 0;
    private int highestlength = 0;
    public final int WIDTH;
    public final int HEIGHT;
    private boolean whileloop = true;

    public Snake(int rows, int cols, int unitsize) {
        WIDTH = cols;
        HEIGHT = rows;
        int height = unitsize * rows;
        int width = unitsize * cols;
        grid = new Grid(rows, cols, height, width);
        Scene scene = new Scene(grid, width, height);
        this.setScene(scene);
        this.setResizable(false);
        this.sizeToScene();
        scene.setOnKeyPressed((t) -> keys.put(t.getCode(), true));
        newGame();
    }

    private void update() {
        externalupdate.run();
        if (isSnakeAlive() && !isGameWon()) {
            int[] vector = null;
            if (keys.getOrDefault(up, false)) {
                vector = new int[]{0, -1};
                keys.put(up, false);
            } else if (keys.getOrDefault(down, false)) {
                vector = new int[]{0, 1};
                keys.put(down, false);
            } else if (keys.getOrDefault(left, false)) {
                vector = new int[]{-1, 0};
                keys.put(left, false);
            } else if (keys.getOrDefault(right, false)) {
                vector = new int[]{1, 0};
                keys.put(right, false);
            }
            if (vector != null) {
                int[] oppositevector = vectorOpposite(vector);
                if (oppositevector[0] != snakedirection[0] || oppositevector[1] != snakedirection[1]) {
                    snakedirection = vector;
                }
            }
            event = grid.moveSnake(snakedirection);
        } else {
            if (highestlength < grid.getSnakeLength()) {
                highestlength = grid.getSnakeLength();
            }
            completedgames++;
            newGame();
        }
    }

    private void newGame() {
        grid.clear();
        grid.newSnake(STARTDIRECTION);
        grid.newApple();
        snakedirection = vectorOpposite(STARTDIRECTION);
        event = 0;
    }

    private static int[] vectorOpposite(int[] vector) {
        return new int[]{-1 * vector[0], -1 * vector[1]};
    }

    public void play() {
        if (whileloop) {
            while (true) {
                update();
            }
        } else {
            this.show();
            loop.setCycleCount(-1);
            loop.play();
        }
    }

    public void setHighScoreReset() {
        highestlength = 0;
    }

    public void setGameSpeed(double speed) {
        loop.setRate(speed);
    }

    public void setKeyUp(KeyCode key) {
        up = key;
    }

    public void setKeyDown(KeyCode key) {
        down = key;
    }

    public void setKeyLeft(KeyCode key) {
        left = key;
    }

    public void setKeyRight(KeyCode key) {
        right = key;
    }

    public void setUpdate(Runnable update) {
        externalupdate = update;
    }

    public void setInput(KeyCode key) {
        keys.put(key, true);
    }

    public void setInput(int action) {
        if (action == 0) {
            keys.put(up, true);
        } else if (action == 1) {
            keys.put(down, true);
        } else if (action == 2) {
            keys.put(left, true);
        } else if (action == 3) {
            keys.put(right, true);
        }
    }

    public float[][] getGameState() {
        return grid.grid;
    }

    public boolean isGameOver() {
        return (event & 1) == 1;
    }

    public boolean isSnakeAlive() {
        return (event & 1) != 1;
    }

    public boolean isAppleEaten() {
        return ((event >> 1) & 1) == 1;
    }

    public boolean isGameWon() {
        return ((event >> 2) & 1) == 1;
    }

    public int getGamesCount() {
        return completedgames;
    }

    public int getHighestLength() {
        return highestlength;
    }

    public int getLength() {
        return grid.snakelength;
    }

    public int[] getHeadLoc() {
        return grid.getHeadLoc();
    }

    public class Grid extends Group {

        private float[][] grid;
        private final int cols;
        private final int rows;
        private final int unitheight;
        private final int unitwidth;
        private ArrayList<int[]> snake = new ArrayList<>();//{int x, int y, int indexInGroupChildren};
        private int snakelength = 0;
        private final int HEAD = -10;
        private final int BODY = -5;
        private final int APPLE = 10;
        private final Group snakesprites = new Group();
        private final Group applesprites = new Group();
        private final int SNAKEGROWTH = 1;
        private final int SNAKESTARTLENGTH = 1;

        public Grid(int rows, int cols, int height, int width) {
            Rectangle background = new Rectangle(width, height, Sprites.BACKCOLOR);
            this.getChildren().addAll(background, snakesprites, applesprites);
            grid = new float[rows][cols];
            this.rows = rows;
            this.cols = cols;
            unitheight = height / rows;
            unitwidth = width / cols;
        }

        public void clear() {
            grid = new float[rows][cols];
            snake.clear();
            snakelength = 0;
            snakesprites.getChildren().clear();
            applesprites.getChildren().clear();
        }

        public void newSnake(int[] taildirectionvector) {
            int componentx = taildirectionvector[0];
            int componenty = taildirectionvector[1];
            int y = rows / 2;
            int x = cols / 2;
            newSnakePart(HEAD, x, y);
            for (int bodycount = 0; bodycount < SNAKESTARTLENGTH; bodycount++) {
                x += componentx;
                y += componenty;
                newSnakePart(BODY, x, y);
            }
        }

        public void newApple() {
            int randomx = (int) (Math.random() * cols);
            int randomy = (int) (Math.random() * rows);
            boolean overlap;
            while (true) {
                overlap = false;
                for (int i = 0; i < snakelength; i++) {
                    int[] snakepart = snake.get(i);
                    if (randomx == snakepart[0] && randomy == snakepart[1]) {
                        overlap = true;
                        break;
                    }
                }
                if (!overlap) {
                    break;
                }
                randomx = (int) (Math.random() * cols);
                randomy = (int) (Math.random() * rows);
            }
            ImageView apple = new ImageView(Sprites.APPLE);
            apple.setFitHeight(unitheight);
            apple.setFitWidth(unitwidth);
            setNodeLoc(apple, randomx, randomy);
            applesprites.getChildren().add(apple);
            grid[randomy][randomx] = APPLE;
        }

        public int moveSnake(int[] vector) {
            int event = 0;
            int componentx = vector[0];
            int componenty = vector[1];
            int[] snakehead = snake.get(0);
            int prevx = snakehead[0];
            int prevy = snakehead[1];
            snakehead[0] += componentx;
            snakehead[1] += componenty;
            int newx = snakehead[0];
            int newy = snakehead[1];
            if (newx < 0 || newx >= cols || newy < 0 || newy >= rows) {
                return 1;
            }
            setNodeLoc(snakesprites.getChildren().get(snakehead[2]), newx, newy);
            int tailindex = snakelength - 1;
            for (int i = 1; i < tailindex; i++) {
                int[] segment = snake.get(i);
                setNodeLoc(snakesprites.getChildren().get(segment[2]), prevx, prevy);
                int segmentx = segment[0];
                int segmenty = segment[1];
                grid[prevy][prevx] = BODY;
                segment[0] = prevx;
                segment[1] = prevy;
                prevx = segmentx;
                prevy = segmenty;
            }
            int[] segment = snake.get(tailindex);
            setNodeLoc(snakesprites.getChildren().get(segment[2]), prevx, prevy);
            int segmentx = segment[0];
            int segmenty = segment[1];
            grid[prevy][prevx] = BODY;
            if (prevy != segmenty || prevx != segmentx) {
                grid[segmenty][segmentx] = 0;
            }
            segment[0] = prevx;
            segment[1] = prevy;
            prevx = segmentx;
            prevy = segmenty;
            if (grid[newy][newx] == BODY) {
                return 1;
            }
            if (grid[newy][newx] == APPLE) {
                event = event | (1 << 1);
                for (int i = 0; i < SNAKEGROWTH; i++) {
                    int[] lastpart = snake.get(snakelength - 1);
                    newSnakePart(BODY, lastpart[0], lastpart[1]);
                }
                applesprites.getChildren().clear();
                if (!isGridFilled()) {
                    newApple();
                } else {
                    event = event | (1 << 2);
                }
            }
            grid[newy][newx] = HEAD;
            return event;
        }

        private void newSnakePart(int type, int x, int y) {
            ImageView segmentimage;
            if (type == HEAD) {
                segmentimage = new ImageView(Sprites.HEAD);
            } else {
                segmentimage = new ImageView(Sprites.BODY);
            }
            segmentimage.setFitHeight(unitheight);
            segmentimage.setFitWidth(unitwidth);
            setNodeLoc(segmentimage, x, y);
            snakesprites.getChildren().add(segmentimage);
            snake.add(new int[]{x, y, snakesprites.getChildren().size() - 1});
            snakelength++;
        }

        private void setNodeLoc(Node node, int x, int y) {
            node.setTranslateX(x * unitwidth);
            node.setTranslateY(y * unitheight);
        }

        private boolean isGridFilled() {//Must be called after apple is removed
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (grid[i][j] == 0) {
                        return false;
                    }
                }
            }
            return true;
        }

        public int getSnakeLength() {
            return snakelength;
        }

        public int[] getHeadLoc() {
            return new int[]{snake.get(0)[0], snake.get(0)[1]};
        }
    }

    public static class Sprites {

        public static final WritableImage APPLE;
        public static final WritableImage HEAD;
        public static final WritableImage BODY;
        public static final WritableImage EMPTY;
        public static final int RESOLUTION = 32;
        public static final Color BACKCOLOR = Color.web("151515");
        public static final Color APPLECOLOR = Color.CRIMSON;
        public static final Color HEADCOLOR = Color.YELLOWGREEN;
        public static final Color BODYCOLOR = Color.SEAGREEN;

        static {
            APPLE = getImageApple();
            HEAD = getImageHead();
            BODY = getImageBody();
            EMPTY = getImageEmpty();
        }

        private static WritableImage getImageApple() {
            int bordersize = RESOLUTION / 8;
            WritableImage image = new WritableImage(RESOLUTION, RESOLUTION);
            PixelWriter pw = image.getPixelWriter();
            for (int y = 0; y < RESOLUTION; y++) {
                for (int x = 0; x < RESOLUTION; x++) {
                    if (y < bordersize || x < bordersize || y >= RESOLUTION - bordersize || x >= RESOLUTION - bordersize) {
                        pw.setColor(x, y, BACKCOLOR);
                    } else {
                        pw.setColor(x, y, APPLECOLOR);
                    }
                }
            }
            return image;
        }

        private static WritableImage getImageHead() {
            int bordersize = RESOLUTION / 8;
            WritableImage image = new WritableImage(RESOLUTION, RESOLUTION);
            PixelWriter pw = image.getPixelWriter();
            for (int y = 0; y < RESOLUTION; y++) {
                for (int x = 0; x < RESOLUTION; x++) {
                    if (y < bordersize || x < bordersize || y >= RESOLUTION - bordersize || x >= RESOLUTION - bordersize) {
                        pw.setColor(x, y, BACKCOLOR);
                    } else {
                        pw.setColor(x, y, HEADCOLOR);
                    }
                }
            }
            return image;
        }

        private static WritableImage getImageBody() {
            int bordersize = RESOLUTION / 8;
            WritableImage image = new WritableImage(RESOLUTION, RESOLUTION);
            PixelWriter pw = image.getPixelWriter();
            for (int y = 0; y < RESOLUTION; y++) {
                for (int x = 0; x < RESOLUTION; x++) {
                    if (y < bordersize || x < bordersize || y >= RESOLUTION - bordersize || x >= RESOLUTION - bordersize) {
                        pw.setColor(x, y, BACKCOLOR);
                    } else {
                        pw.setColor(x, y, BODYCOLOR);
                    }
                }
            }
            return image;
        }

        private static WritableImage getImageEmpty() {
            WritableImage image = new WritableImage(RESOLUTION, RESOLUTION);
            PixelWriter pw = image.getPixelWriter();
            for (int y = 0; y < RESOLUTION; y++) {
                for (int x = 0; x < RESOLUTION; x++) {
                    pw.setColor(x, y, BACKCOLOR);
                }
            }
            return image;
        }
    }
}
