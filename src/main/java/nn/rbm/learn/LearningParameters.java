package nn.rbm.learn;

import com.google.common.base.Function;
import math.functions.Sigmoid;

/**
 * Created by kenny on 5/15/14.
 */
public class LearningParameters {

    private double learningRate = 0.1;

    private Function<Double, Double> logisticsFunction = new Sigmoid();

    private int epochs = 15000;

    public double getLearningRate() {
        return learningRate;
    }

    public LearningParameters setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public Function<Double, Double> getLogisticsFunction() {
        return logisticsFunction;
    }

    public LearningParameters setLogisticsFunction(Function<Double, Double> logisticsFunction) {
        this.logisticsFunction = logisticsFunction;
        return this;
    }

    public int getEpochs() {
        return epochs;
    }

    public LearningParameters setEpochs(int epochs) {
        this.epochs = epochs;
        return this;
    }

    @Override
    public String toString() {
        return "TrainingParameters{" +
                "learningRate=" + learningRate +
                ", logisticsFunction=" + logisticsFunction +
                ", epochs=" + epochs +
                '}';
    }
}
