package nn.rbm;

import data.image.Image;
import data.image.decode.Matrix24BitImageDecoder;
import data.image.encode.Matrix24BitImageEncoder;
import math.matrix.Matrix;
import nn.rbm.deep.DeepRBM;
import nn.rbm.deep.LayerParameters;
import nn.rbm.factory.RandomRBMFactory;
import nn.rbm.learn.DeepContrastiveDivergence;
import nn.rbm.learn.LearningParameters;
import nn.rbm.learn.MultiThreadedDeepContrastiveDivergence;
import org.apache.log4j.Logger;
import org.junit.Test;
import utils.Clock;

/**
 * Created by kenny on 5/12/14.
 */
public class TestThreadedRBM {

    private static final Logger LOGGER = Logger.getLogger(TestThreadedRBM.class);

    private static final RandomRBMFactory RBM_FACTORY = new RandomRBMFactory();

    @Test
    public void singleVsMultiThread() {
        // 100 * 63 * 24 input (151200)
        final LayerParameters[] layerParameters = new LayerParameters[] {
                new LayerParameters().setNumRBMS(200).setVisibleUnitsPerRBM(756).setHiddenUnitsPerRBM(100),    // 151,200 in, 20,000 out
                new LayerParameters().setNumRBMS(100).setVisibleUnitsPerRBM(200).setHiddenUnitsPerRBM(50),     // 20,000 in, 5,000 out
                new LayerParameters().setNumRBMS(50).setVisibleUnitsPerRBM(100).setHiddenUnitsPerRBM(10),     // 5,000 in, 500 out
                new LayerParameters().setNumRBMS(25).setVisibleUnitsPerRBM(20).setHiddenUnitsPerRBM(10),     // 500 in, 250 out
                new LayerParameters().setNumRBMS(10).setVisibleUnitsPerRBM(25).setHiddenUnitsPerRBM(5),    // 250 in, 50 out
                new LayerParameters().setNumRBMS(1).setVisibleUnitsPerRBM(50).setHiddenUnitsPerRBM(100),    // 50 in, 100 out
        };

        final DeepRBM deepRBM = new DeepRBM(layerParameters, RBM_FACTORY);
        final LearningParameters learningParameters = new LearningParameters().setEpochs(10).setLog(false);
        final DeepContrastiveDivergence deepContrastiveDivergence = new DeepContrastiveDivergence(learningParameters);
        final MultiThreadedDeepContrastiveDivergence multiThreadedDeepContrastiveDivergence = new MultiThreadedDeepContrastiveDivergence(learningParameters);

        final Image jetImage = new Image("/data/fighter_jet_small.jpg");
        final Matrix jetMatrix = new Matrix24BitImageEncoder().encode(jetImage);

        LOGGER.info("Start training");
        // time the training
        final Clock clock = new Clock();
        clock.start();
        deepContrastiveDivergence.learn(deepRBM, jetMatrix);
        final long time1 = clock.elapsedMillis();

        clock.reset();
        multiThreadedDeepContrastiveDivergence.learn(deepRBM, jetMatrix);
        final long time2 = clock.elapsedMillis();
        LOGGER.info("Single Threaded Time: " + time1 + "ms");  // 100 epocs avg 69,934ms, 10 epocs avg 7765ms
        LOGGER.info("Multi Threaded Time: " + time2 + "ms");   // 100 epocs avg 36,188ms, 10 epocs avg 3637ms

        final Matrix hidden = deepContrastiveDivergence.runVisible(deepRBM, jetMatrix);
        final Matrix visual = deepContrastiveDivergence.runHidden(deepRBM, hidden);
        final Image outImage = new Matrix24BitImageDecoder(63).decode(visual); // 19/63/250
        outImage.save("/tmp/fighter_rendered_small_24bit_deep_singlethread.bmp");

        final Matrix hidden2 = multiThreadedDeepContrastiveDivergence.runVisible(deepRBM, jetMatrix);
        final Matrix visual2 = multiThreadedDeepContrastiveDivergence.runHidden(deepRBM, hidden2);
        final Image outImage2 = new Matrix24BitImageDecoder(63).decode(visual2); // 19/63/250
        outImage2.save("/tmp/fighter_rendered_small_24bit_deep_multithread.bmp");
    }

}
