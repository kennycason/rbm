package nn.rbm.deep;

import nn.rbm.RBM;

import java.util.Arrays;

/**
 * Created by kenny on 5/16/14.
 */
public class RBMLayer {

    public final RBM[] rbms;

    public RBMLayer(RBM[] rbms) {
        this.rbms = rbms;
    }

    public RBM getRBM(int r) {
        return rbms[r];
    }

    public int size() {
        return rbms.length;
    }

    @Override
    public String toString() {
        return "RBMLayer{" +
                "rbms=" + Arrays.toString(rbms) +
                '}';
    }
}

