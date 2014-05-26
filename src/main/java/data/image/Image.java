package data.image;

import org.apache.log4j.Logger;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * Created by kenny on 5/20/14.
 */
public class Image {

    private static final Logger LOGGER = Logger.getLogger(Image.class);

    private BufferedImage bi;

    public Image(String file) {
        try {
            bi = ImageIO.read(Image.class.getResourceAsStream(file));
        } catch (IOException e) {
            LOGGER.info(e.getMessage(), e);
        }
    }

    public Image(final BufferedImage bi) {
        this.bi = bi;
    }

    public void set(int x, int y, int rgb) {
        bi.setRGB(x, y, rgb);
    }

    public int get(int x, int y) {
        return bi.getRGB(x, y);
    }

    public BufferedImage data() {
        return bi;
    }

    public int width() {
        return bi.getWidth();
    }

    public int height() {
        return bi.getHeight();
    }

    private static String getFormat(String file) {
        String[] parts =  file.split("\\.");
        return parts[parts.length - 1];
    }

    public void save(String file) {
        try {
            LOGGER.info("Writing file: " + file);
            ImageIO.write(bi, getFormat(file), new File(file));
        } catch (IOException e) {
            LOGGER.error(e.getMessage(), e);
        }
    }

    public void saveThumbnail(String file, double scale) {
        int width = (int) (bi.getWidth() * scale);
        int height = (int) (bi.getHeight() * scale);

        int imgWidth = bi.getWidth();
        int imgHeight = bi.getHeight();
        if (imgWidth * height < imgHeight * width) {
            width = imgWidth * height / imgHeight;
        } else {
            height = imgHeight * width / imgWidth;
        }
        BufferedImage newImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = newImage.createGraphics();
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
            g.setBackground(Color.BLACK);
            g.clearRect(0, 0, width, height);
            g.drawImage(bi, 0, 0, width, height, null);
        } finally {
            g.dispose();
        }
        try {
            LOGGER.info("Writing file: " + file);
            ImageIO.write(newImage, getFormat(file), new File(file));
        } catch (IOException e) {
            LOGGER.error(e.getMessage(), e);
        }
    }

}
