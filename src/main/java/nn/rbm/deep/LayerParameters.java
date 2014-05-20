package nn.rbm.deep;

/**
 * Created by kenny on 5/16/14.
 */
public class LayerParameters {

    private int numRBMS = 1;

    private int visibleUnitsPerRBM = 1;

    private int hiddenUnitsPerRBM = 1;

    public int getNumRBMS() {
        return numRBMS;
    }

    public LayerParameters setNumRBMS(int numRBMS) {
        this.numRBMS = numRBMS;
        return this;
    }

    public int getVisibleUnitsPerRBM() {
        return visibleUnitsPerRBM;
    }

    public LayerParameters setVisibleUnitsPerRBM(int visibleUnitsPerRBM) {
        this.visibleUnitsPerRBM = visibleUnitsPerRBM;
        return this;
    }

    public int getHiddenUnitsPerRBM() {
        return hiddenUnitsPerRBM;
    }

    public LayerParameters setHiddenUnitsPerRBM(int hiddenUnitsPerRBM) {
        this.hiddenUnitsPerRBM = hiddenUnitsPerRBM;
        return this;
    }
}
