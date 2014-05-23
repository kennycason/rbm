package utils.concurrent;

/**
 * Created by kenny on 5/22/14.
 */
import org.apache.log4j.Logger;

import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ThreadPoolExecutor {
    private static final Logger LOGGER = Logger.getLogger(ThreadPoolExecutor.class);

    public static final int DEFAULT_NUMBER_THREADS = 10;

    private int numThreads = DEFAULT_NUMBER_THREADS;

    private List<Callable<Boolean>> callables = new LinkedList<>();

    public int getNumThreads() {
        return numThreads;
    }

    public void setNumThreads(int numThreads) {
        this.numThreads = numThreads;
    }

    public void add(Runnable r) {
        callables.add(new PoolWorker(r));
    }

    public void clearAllRunners() {
        this.callables.clear();
    }

    public void execute() {
        LOGGER.debug("Thread pool launched with: " + numThreads + " threads.");
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        try {
            executor.invokeAll(callables);
        } catch (InterruptedException e) {
            LOGGER.error(e.getMessage(), e);
        }
        finally {
            LOGGER.debug("Thread Pool Finished, shutting down");
            executor.shutdown();
            clearAllRunners();
        }

    }

    public void addAll(Collection<? extends Runnable> runners) {
        for (Runnable runner : runners) {
            this.add(runner);
        }
    }


    private class PoolWorker implements Callable<Boolean> {

        private Runnable r;

        public PoolWorker(Runnable r) {
            this.r = r;
        }

        @Override
        public Boolean call() throws Exception {
            r.run();
            return true;
        }

    }

}
