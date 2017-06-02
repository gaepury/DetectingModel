package test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Random Forest
 * 
 */
public class RandomForest {
	public static ArrayList risk = new ArrayList();
	/** the number of threads to use when generating the forest */
	private static final int NUM_THREADS = Runtime.getRuntime().availableProcessors();
	// private static final int NUM_THREADS=2;
	/**
	 * the number of categorical responses of the data (the classes, the "Y"
	 * values) - set this before beginning the forest creation
	 */
	public static int C;
	/**
	 * the number of attributes in the data - set this before beginning the
	 * forest creation
	 */
	public static int M;
	/**
	 * Of the M total attributes, the random forest computation requires a
	 * subset of them to be used and picked via random selection. "Ms" is the
	 * number of attributes in this subset. The formula used to generate Ms was
	 * recommended on Breiman's website.
	 */
	public static int Ms;
	/** the collection of the forest's decision trees */
	private ArrayList<DTree> trees;
	/** the number of trees in this random tree */
	private int numTrees;

	/** This holds all of the predictions of trees in a Forest */
	private ArrayList<ArrayList<Integer>> Prediction;
	
	
	//실시간 검지용--------------
	private int[] realtime_testdata;
	private ArrayList<Integer> realtime_Prediction;
	public static final int RISK_SITUATION = 1;
	public static final int NORMAL_SITUATION = 2;
	//----------------------------------------
	
	
	
	/** the thread pool that controls the generation of the decision trees */
	private ExecutorService treePool;
	/**
	 * the original training data matrix that will be used to generate the
	 * random forest classifier
	 */
	private ArrayList<int[]> data;
	/** the data on which produced random forest will be tested */
	private ArrayList<int[]> testdata;
	//실시간 검지용
	
	
	
	private void creatLogRF(int numTrees, ArrayList<int[]> data) {
		System.out.println("creating " + numTrees + " trees in a random Forest. . . ");
		System.out.println("total data size is " + data.size());
		System.out.println("number of attributes " + (data.get(0).length - 1));
		System.out.println("number of selected attributes "
				+ ((int) Math.round(Math.log(data.get(0).length - 1) / Math.log(2) + 1)));
	}
	public RandomForest(int numTrees, ArrayList<int[]> data){
		this.numTrees = numTrees;
		this.data = data;
		creatLogRF(numTrees, data);
		trees = new ArrayList<DTree>(numTrees);
		realtime_Prediction = new ArrayList<>();
	}
	//original
	public RandomForest(int numTrees, ArrayList<int[]> data, ArrayList<int[]> t_data) {
		this.numTrees = numTrees;
		this.data = data;
		this.testdata = t_data;
		trees = new ArrayList<DTree>(numTrees);
		creatLogRF(numTrees, data);
		Prediction = new ArrayList<ArrayList<Integer>>();
	}
	
	
	
	
	public void createRF() {
		System.out.println("Number of threads started : " + NUM_THREADS);
		System.out.println("Running...");
		treePool = Executors.newFixedThreadPool(NUM_THREADS);
		for (int t = 0; t < numTrees; t++) {
			treePool.execute(new CreateTree(data, this, t + 1));
		}
		treePool.shutdown();
		try {
			treePool.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS); // effectively
																			// infinity
		} catch (InterruptedException ignored) {
			System.out.println("interrupted exception in Random Forests");
		}
		System.out.println("Finished tree construction");
	}
	public int realtime_Start(int[] realtime_t_data) {
		this.realtime_testdata=realtime_t_data;
		createRF();
		int result=realtimePrediction(trees, realtime_testdata);
		return result;
	}
	public void Start() {
//		createRF();
		TestForest(trees, testdata);
	}
	
	
	
	private int realtimePrediction(ArrayList<DTree> collec_tree, int[] test_data) {
		if(!realtime_Prediction.isEmpty()){
			realtime_Prediction.clear();
		}
		int treee = 1;
		for (DTree dt : collec_tree) {
			realtime_Prediction.add(dt.realtime_CalculateClass(test_data, treee));
		}
		ArrayList<Integer> Val = new ArrayList<Integer>();
		for (int i = 0; i < collec_tree.size(); i++) {
			Val.add(realtime_Prediction.get(i));
		}
		int pred = ModeOf(Val);
		System.out.print("랜덤포레스트 검지 : [before:"+test_data[0]+", after:"+test_data[1]+", avg_decel:"+test_data[2]+", max_decel:"+test_data[3]+", distance:"+test_data[4]+", second:"+test_data[5]+"]");
		if (pred == 1) {
			System.out.println("-위험");
			return RISK_SITUATION;
		} else {
			System.out.println("-일반");
			return NORMAL_SITUATION;
		}

	}
	private void TestForest(ArrayList<DTree> collec_tree, ArrayList<int[]> test_data) {
		int correstness = 0;
		int k = 0;
		 ArrayList<Integer> ActualValues = new ArrayList<Integer>();
		
		 for (int[] rec : test_data) {
		 ActualValues.add(rec[rec.length - 1]);
		 }
		 System.out.println("ActualClass : "+ActualValues);

		int treee = 1;
		int c = 1;
		for (DTree dt : collec_tree) {

			dt.CalculateClasses(test_data, treee);
			Prediction.add(dt.predictions);
		}

		System.out.println("트리 수:" + Prediction.size());
		System.out.println("데이터 수:" + Prediction.get(0).size());

		String str = "[";
		for (int i = 0; i < test_data.size(); i++) {
			ArrayList<Integer> Val = new ArrayList<Integer>();
			for (int j = 0; j < collec_tree.size(); j++) {
				Val.add(Prediction.get(j).get(i));
			}
			int pred = ModeOf(Val);

			if (pred == 1) {
				risk.add(i);
			}
			if (i != test_data.size() - 1) {
				str += pred + ", ";
			} else {
				str += pred + "]";
			}

			// if (pred == ActualValues.get(i)) {
			// correstness = correstness + 1;
			// }
			// System.out.println(Val);
		}

		System.out.println("PredictionClass : " + str);

		 System.out.println("Accuracy of Forest is : " + (100 * correstness /
		 test_data.size()) + "%");
	}

	
	private int ModeOf(ArrayList<Integer> treePredict) {
		int max = 0, maxclass = -1;
		// TODO Auto-generated method stub
		for (int i = 0; i < treePredict.size(); i++) {
			int count = 0;
			for (int j = 0; j < treePredict.size(); j++) {
				if (treePredict.get(j) == treePredict.get(i)) {
					count++;
				}
				if (count > max) {
					maxclass = treePredict.get(i);
					max = count;
				}
			}
		}
		return maxclass;
	}

	/**
	 * This class houses the machinery to generate one decision tree in a thread
	 * pool environment.
	 * 
	 * @author kapelner
	 *
	 */
	private class CreateTree implements Runnable {
		/**
		 * the training data to generate the decision tree (same for all trees)
		 */
		private ArrayList<int[]> data;
		/** the current forest */
		private RandomForest forest;
		/** the Tree number */
		private int treenum;

		/**
		 * A default, dummy constructor
		 */
		public CreateTree(ArrayList<int[]> data, RandomForest forest, int num) {
			this.data = data;
			this.forest = forest;
			this.treenum = num;
		}

		/**
		 * Creates the decision tree
		 */
		public void run() {
			// System.out.println("Creating a Dtree num : "+treenum+" ");
			trees.add(new DTree(data, forest, treenum));
			// System.out.println("tree added in RandomForest.AddTree.run()");
		}
	}

	/**
	 * Evaluates an incoming data record. It first allows all the decision trees
	 * to classify the record, then it returns the majority vote
	 * 
	 * @param record
	 *            the data record to be classified
	 */
	public int Evaluate(int[] record) {
		int[] counts = new int[C];
		for (int t = 0; t < numTrees; t++) {
			int Class = (trees.get(t)).Evaluate(record);
			counts[Class]++;
		}
		return FindMaxIndex(counts);
	}

	public static int FindMaxIndex(int[] arr) {
		int index = 0;
		int max = Integer.MIN_VALUE;
		for (int i = 0; i < arr.length; i++) {
			if (arr[i] > max) {
				max = arr[i];
				index = i;
			}
		}
		return index;
	}

	/**
	 * Attempt to abort random forest creation
	 */
	public void Stop() {
		treePool.shutdownNow();
	}

}
