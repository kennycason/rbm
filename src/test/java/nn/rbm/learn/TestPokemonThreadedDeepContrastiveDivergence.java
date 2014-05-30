package nn.rbm.learn;

import data.image.Image;
import data.image.decode.Matrix24BitImageDecoder;
import data.image.encode.Matrix24BitIgnoreWhiteImageEncoder;
import math.DenseMatrix;
import math.Matrix;
import nn.rbm.deep.DeepRBM;
import nn.rbm.deep.LayerParameters;
import nn.rbm.factory.RandomRBMFactory;
import nn.rbm.save.DeepRBMPersister;
import org.apache.log4j.Logger;
import org.junit.Test;
import utils.Clock;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by kenny on 5/12/14.
 */
public class TestPokemonThreadedDeepContrastiveDivergence {

    private static final Logger LOGGER = Logger.getLogger(TestPokemonThreadedDeepContrastiveDivergence.class);

    private static final RandomRBMFactory RBM_FACTORY = new RandomRBMFactory();

    private static final DeepRBMPersister DEEP_RBM_PERSISTER = new DeepRBMPersister();

    private static final int POKEMON_COUNT = 25;

    @Test
    public void pokemonTest() {
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

        final Matrix allPokemonData = loadDataSet();

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

    private Matrix loadDataSet() {
        int i = 0;
        final List<String> imageFiles = readImageFolder();
        final Matrix[] allPokemonData = new Matrix[imageFiles.size()];
        for(String imageFile : imageFiles) {
            final Image pokemonImage = new Image(imageFile);
            final Matrix pokemonData = new Matrix24BitIgnoreWhiteImageEncoder().encode(pokemonImage);
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

    private List<String> readFirstPokemon() {
        final List<String> imageFiles = new ArrayList<>(3);
        imageFiles.add("/data/pokemon/input/pokemon_151 [www.imagesplitter.net]-0-0.png");
        imageFiles.add("/data/pokemon/input/pokemon_151 [www.imagesplitter.net]-1-0.png");
        imageFiles.add("/data/pokemon/input/pokemon_151 [www.imagesplitter.net]-2-0.png");

        return imageFiles;
    }

}
