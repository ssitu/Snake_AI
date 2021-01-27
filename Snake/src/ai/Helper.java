package ai;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.RandomAccessFile;
import snake.Snake;

public class Helper {

    public static void writeStatsToCSV(FileWriter fw, Snake game, float avgLength) {
        try {
            fw.append(Integer.toString(game.getGamesPlayed()));
            fw.append(',');
            fw.append(Float.toString(avgLength));
            fw.append('\n');
            fw.flush();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static String lastLine(String path) throws IOException{
        File src = new File(path);
        StringBuilder builder = new StringBuilder();
        RandomAccessFile randomAccessFile = null;
        try {
            randomAccessFile = new RandomAccessFile(src, "r");
            long fileLength = src.length() - 2;
            randomAccessFile.seek(fileLength);
            for (long pointer = fileLength; pointer >= 0; pointer--) {
                randomAccessFile.seek(pointer);
                char c;
                c = (char) randomAccessFile.read();
                if (c == '\n') {
                    break;
                }
                builder.append(c);
            }
            builder.reverse();
            return builder.toString();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.exit(-1);
        } finally {
            if (randomAccessFile != null) {
                try {
                    randomAccessFile.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        return null;
    }
}
