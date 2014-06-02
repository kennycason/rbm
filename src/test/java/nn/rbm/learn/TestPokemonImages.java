package nn.rbm.learn;

import data.image.Image;
import data.image.decode.Matrix24BitImageDecoder;
import data.image.decode.Matrix3BitScaledImageDecoder;
import data.image.encode.Matrix24BitIgnoreRGBImageEncoder;
import data.image.encode.Matrix24BitImageEncoder;
import data.image.encode.Matrix3BitScaledImageEncoder;
import data.image.encode.MatrixImageEncoder;
import math.DenseMatrix;
import math.Matrix;
import nn.rbm.RBM;
import nn.rbm.deep.DeepRBM;
import nn.rbm.deep.LayerParameters;
import nn.rbm.factory.RandomRBMFactory;
import nn.rbm.save.DeepRBMPersister;
import nn.rbm.save.RBMPersister;
import org.apache.log4j.Logger;
import org.junit.Test;
import utils.Clock;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by kenny on 5/12/14.
 */
public class TestPokemonImages {

    private static final Logger LOGGER = Logger.getLogger(TestPokemonImages.class);

    private static final RandomRBMFactory RBM_FACTORY = new RandomRBMFactory();

    private static final DeepRBMPersister DEEP_RBM_PERSISTER = new DeepRBMPersister();

    private static final RBMPersister RBM_PERSISTER = new RBMPersister();

    private static final int POKEMON_COUNT = 25;

    @Test
    public void pokemonTest() {
        // fails with particularly deep rbms
        // 60 * 60 * 24 input (86,400)
//        final LayerParameters[] layerParameters = new LayerParameters[] {
//                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(86400).setHiddenUnitsPerRBM(1000),
//         //       new LayerParameters().setNumRBMS(200).setVisibleUnitsPerRBM(432).setHiddenUnitsPerRBM(100),    // 86,400 in, 20,000 out
//           //     new LayerParameters().setNumRBMS(100).setVisibleUnitsPerRBM(200).setHiddenUnitsPerRBM(50),     // 20,000 in, 5,000 out
//           //     new LayerParameters().setNumRBMS(50).setVisibleUnitsPerRBM(100).setHiddenUnitsPerRBM(10),     // 5,000 in, 500 out
////                new LayerParameters().setNumRBMS(25).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(10),     // 500 in, 250 out
////                new LayerParameters().setNumRBMS(10).setVisibleUnitsPerRBM(25).setHiddenUnitsPerRBM(5),    // 250 in, 50 out
////                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(50).setHiddenUnitsPerRBM(5000),    // 50 in, 100 out
//        };

        final LayerParameters[] layerParameters = new LayerParameters[] {
                new LayerParameters().setNumRBMS(2).setVisibleUnitsPerRBM(43200).setHiddenUnitsPerRBM(1000),    // 86,400 in, 2,000 out
                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(2000).setHiddenUnitsPerRBM(1000),    // 2000 in, 1000 out
        };

        final DeepRBM deepRBM = new DeepRBM(layerParameters, RBM_FACTORY);
        final LearningParameters learningParameters = new LearningParameters().setEpochs(100).setLearningRate(0.1).setLog(true);
        final MultiThreadedDeepContrastiveDivergence multiThreadedDeepContrastiveDivergence = new MultiThreadedDeepContrastiveDivergence(learningParameters, 4);

        final Matrix allPokemonData = loadDataSet(readImageFolder(), new Matrix24BitIgnoreRGBImageEncoder(0xFFFFFF));

        LOGGER.info("Start training");
        final Clock clock = new Clock();
        clock.start();

        multiThreadedDeepContrastiveDivergence.learn(deepRBM, allPokemonData);

        final long time = clock.elapsedMillis();
        LOGGER.info(allPokemonData.rows() + " Pokemon learned in: " + time + "ms, " + (time / 1000.0 / 60.0) + "min");

        DEEP_RBM_PERSISTER.save(deepRBM, "/tmp/rbm_pokemon.csv");

        for(int i = 0; i < allPokemonData.rows(); i++) {
            final Matrix pokemon = DenseMatrix.make(allPokemonData.row(i));
            final Matrix hidden = multiThreadedDeepContrastiveDivergence.runVisible(deepRBM, pokemon);
            final Matrix visual = multiThreadedDeepContrastiveDivergence.runHidden(deepRBM, hidden);
            final Image outImage = new Matrix24BitImageDecoder(60).decode(visual);
            outImage.save("/tmp/pokemon_" + i + ".bmp");
        }

    }

    /**
     * note how having a white BG (max value input) negatively affects learning,
     * where as a Black (zero value input) converges quickly.
     */
    @Test
    public void pokemonBlackVsWhiteBGTest() {
        final RBM rbm = RBM_FACTORY.build(86_400, 20);

        final LearningParameters learningParameters = new LearningParameters().setEpochs(1000).setLearningRate(0.1).setLog(true);
        final ContrastiveDivergence contrastiveDivergence = new ContrastiveDivergence(learningParameters);

        final Matrix whiteBGPokemon = loadDataSet(readFirstPokemon(), new Matrix24BitImageEncoder());
        final Matrix blackBGPokemon = loadDataSet(readFirstPokemon(), new Matrix24BitIgnoreRGBImageEncoder(0xFFFFFF));

        LOGGER.info("Start training White BG Pokemon");
        final Clock clock = new Clock();
        clock.start();
        contrastiveDivergence.learn(rbm, whiteBGPokemon);
        final long time = clock.elapsedMillis();
        LOGGER.info(whiteBGPokemon.rows() + " Pokemon learned in: " + time + "ms, " + (time / 1000.0 / 60.0) + "min");
        RBM_PERSISTER.save(rbm, "/tmp/rbm_pokemon_white_bg.csv");

        LOGGER.info("Start training Black BG Pokemon");
        clock.reset();
        contrastiveDivergence.learn(rbm, blackBGPokemon);
        final long time2 = clock.elapsedMillis();
        LOGGER.info(blackBGPokemon.rows() + " Pokemon learned in: " + time2 + "ms, " + (time2/ 1000.0 / 60.0) + "min");
        RBM_PERSISTER.save(rbm, "/tmp/rbm_pokemon_black_bg.csv");

        for(int i = 0; i < whiteBGPokemon.rows(); i++) {
            final Matrix pokemon = DenseMatrix.make(whiteBGPokemon.row(i));
            final Matrix hidden = contrastiveDivergence.runVisible(rbm, pokemon);
            final Matrix visual = contrastiveDivergence.runHidden(rbm, hidden);
            final Image outImage = new Matrix24BitImageDecoder(60).decode(visual);
            outImage.save("/tmp/pokemon_white_bg_" + i + ".bmp");
        }

        for(int i = 0; i < blackBGPokemon.rows(); i++) {
            final Matrix pokemon = DenseMatrix.make(blackBGPokemon.row(i));
            final Matrix hidden = contrastiveDivergence.runVisible(rbm, pokemon);
            final Matrix visual = contrastiveDivergence.runHidden(rbm, hidden);
            final Image outImage = new Matrix24BitImageDecoder(60).decode(visual);
            outImage.save("/tmp/pokemon_black_bg_" + i + ".bmp");
        }

    }


    @Test
    public void pokemon3BitScaledColorScheme() {
        // 60 * 60 * 3
        final RBM rbm = RBM_FACTORY.build(108_00, 40);

        final LearningParameters learningParameters = new LearningParameters().setEpochs(1000).setLearningRate(0.1).setLog(true);
        final ContrastiveDivergence contrastiveDivergence = new ContrastiveDivergence(learningParameters);

        final Matrix firstPokemon = loadDataSet(readFirstPokemon(), new Matrix3BitScaledImageEncoder(0xFFFFFF));

        LOGGER.info("Start training");
        final Clock clock = new Clock();
        clock.start();
        contrastiveDivergence.learn(rbm, firstPokemon);
        final long time = clock.elapsedMillis();
        LOGGER.info(firstPokemon.rows() + " Pokemon learned in: " + time + "ms, " + (time / 1000.0 / 60.0) + "min");
        RBM_PERSISTER.save(rbm, "/tmp/rbm_pokemon_scaled_rgb.csv");

        for(int i = 0; i < firstPokemon.rows(); i++) {
            final Matrix pokemon = DenseMatrix.make(firstPokemon.row(i));
            final Matrix hidden = contrastiveDivergence.runVisible(rbm, pokemon);
            final Matrix visual = contrastiveDivergence.runHidden(rbm, hidden);
            final Image outImage = new Matrix3BitScaledImageDecoder(60).decode(visual);
            outImage.save("/tmp/pokemon_scaled_rgb_" + i + ".bmp");
        }

    }


    /**
     * train pokemon images then iterate over each feature and see what is rendered
     * TODO iterate over various sets of features
     */
    @Test
    public void pokemonFeatureTest() {
        final RBM rbm = RBM_FACTORY.build(86_400, 20);

        final LearningParameters learningParameters = new LearningParameters().setEpochs(1000).setLearningRate(0.1).setLog(true);
        final ContrastiveDivergence contrastiveDivergence = new ContrastiveDivergence(learningParameters);

        final Matrix allPokemonData = loadDataSet(readFirstPokemon(), new Matrix24BitIgnoreRGBImageEncoder(0xFFFFFF));

        LOGGER.info("Start training");
        final Clock clock = new Clock();
        clock.start();

        contrastiveDivergence.learn(rbm, allPokemonData);

        final long time = clock.elapsedMillis();
        LOGGER.info(allPokemonData.rows() + " Pokemon learned in: " + time + "ms, " + (time / 1000.0 / 60.0) + "min");

        RBM_PERSISTER.save(rbm, "/tmp/rbm_pokemon.csv");

        for(int i = 0; i < allPokemonData.rows(); i++) {
            final Matrix pokemon = DenseMatrix.make(allPokemonData.row(i));
            final Matrix hidden = contrastiveDivergence.runVisible(rbm, pokemon);
            final Matrix visual = contrastiveDivergence.runHidden(rbm, hidden);
            final Image outImage = new Matrix24BitImageDecoder(60).decode(visual);
            outImage.save("/tmp/pokemon_rbm_" + i + ".bmp");
        }


        final List<Matrix> features = new ArrayList<>();
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{1,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,1,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,1,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,1,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,1, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 1,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 0,1,0,0,0, 0,0,0,0,0, 0,0,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 0,0,1,0,0, 0,0,0,0,0, 0,0,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 0,0,0,1,0, 0,0,0,0,0, 0,0,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 0,0,0,0,1, 0,0,0,0,0, 0,0,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 0,0,0,0,0, 1,0,0,0,0, 0,0,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 0,0,0,0,0, 0,1,0,0,0, 0,0,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 0,0,0,0,0, 0,0,1,0,0, 0,0,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 0,0,0,0,0, 0,0,0,1,0, 0,0,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,1, 0,0,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 1,0,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,1,0,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,1,0,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,1,0}}));
        features.add(DenseMatrix.make(new double[][] {{0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,1}}));
        features.add(DenseMatrix.make(new double[][] {{1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1}}));

        int i = 0;
        for(Matrix feature : features) {
            final Matrix visual = contrastiveDivergence.runHidden(rbm, feature);
            final Image outImage = new Matrix24BitImageDecoder(63).decode(visual); // 19/63/250
            outImage.save("/tmp/pokemon_rbm_feature_" + i + ".bmp");
            i++;
        }
    }

    private Matrix loadDataSet(final List<String> imageFiles, final MatrixImageEncoder encoder) {
        int i = 0;
        final Matrix[] allPokemonData = new Matrix[imageFiles.size()];
        for(String imageFile : imageFiles) {
            final Image pokemonImage = new Image(imageFile);
            final Matrix pokemonData = encoder.encode(pokemonImage);
            allPokemonData[i] = pokemonData;

            i++;
        }
        return DenseMatrix.make(Matrix.concatRows(allPokemonData));
    }

    private List<String> readImageFolder() {
        final List<String> imageFiles = new ArrayList<>(POKEMON_COUNT);
        int p = 0;
        for(int i = 0; i < 11; i++ ) {
            for(int j = 0; j < 15; j++ ) {
                imageFiles.add("/data/pokemon/input/pokemon_151 [www.imagesplitter.net]-" + i + "-" + j + ".png");
                p++;
                if(p == POKEMON_COUNT) { return imageFiles; }
            }
        }

        return Collections.emptyList();
    }

    private List<String> read3Pokemon() {
        final List<String> imageFiles = new ArrayList<>(3);
        imageFiles.add("/data/pokemon/input/pokemon_151 [www.imagesplitter.net]-0-0.png");
        imageFiles.add("/data/pokemon/input/pokemon_151 [www.imagesplitter.net]-1-0.png");
        imageFiles.add("/data/pokemon/input/pokemon_151 [www.imagesplitter.net]-2-0.png");

        return imageFiles;
    }

    private List<String> readFirstPokemon() {
        final List<String> imageFiles = new ArrayList<>(3);
        imageFiles.add("/data/pokemon/input/pokemon_151 [www.imagesplitter.net]-0-0.png");
        imageFiles.add("/data/pokemon/input/pokemon_151 [www.imagesplitter.net]-0-1.png");
        imageFiles.add("/data/pokemon/input/pokemon_151 [www.imagesplitter.net]-0-2.png");
        imageFiles.add("/data/pokemon/input/pokemon_151 [www.imagesplitter.net]-0-3.png");
        imageFiles.add("/data/pokemon/input/pokemon_151 [www.imagesplitter.net]-0-4.png");
        imageFiles.add("/data/pokemon/input/pokemon_151 [www.imagesplitter.net]-0-5.png");
        imageFiles.add("/data/pokemon/input/pokemon_151 [www.imagesplitter.net]-0-6.png");
        imageFiles.add("/data/pokemon/input/pokemon_151 [www.imagesplitter.net]-0-7.png");
        imageFiles.add("/data/pokemon/input/pokemon_151 [www.imagesplitter.net]-0-8.png");
        return imageFiles;
    }

}
