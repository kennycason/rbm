package nn.rbm.learn;

import cern.colt.function.tdouble.DoubleFunction;
import math.functions.Sigmoid;

/**
 * Created by kenny on 5/15/14.
 */
public class LearningParameters {

    private double learningRate = 0.1;

    private DoubleFunction logisticsFunction = new Sigmoid();

    private int epochs = 15000;

    private boolean log = true;

    private int memory = 1;

    public double getLearningRate() {
        return learningRate;
    }

    public LearningParameters setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public DoubleFunction getLogisticsFunction() {
        return logisticsFunction;
    }

    public LearningParameters setLogisticsFunction(DoubleFunction logisticsFunction) {
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


    public boolean isLog() {
        return log;
    }

    public LearningParameters setLog(boolean log) {
        this.log = log;
        return this;
    }

    public int getMemory() {
        return memory;
    }

    public LearningParameters setMemory(int memory) {
        this.memory = memory;
        return this;
    }

    @Override
    public String toString() {
        return "LearningParameters{" +
                "learningRate=" + learningRate +
                ", logisticsFunction=" + logisticsFunction +
                ", epochs=" + epochs +
                ", log=" + log +
                ", memory=" + memory +
                '}';
    }

}
