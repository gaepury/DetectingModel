package test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.LineNumberReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This example is intended to be a simple CSV classifier that seperates the
 * training data from the test data for the classification of animals. It would
 * be suitable as a beginner's example because not only does it load CSV data
 * into the network, it also shows how to extract the data and display the
 * results of the classification, as well as a simple method to map the lables
 * from the testing data into the results.
 *
 * @author Clay Graham
 */

public class dp_rf_compare {

	private static Logger log = LoggerFactory.getLogger(dp_rf_compare.class);
	public static DataNormalization normalizer = new NormalizerStandardize();
	private static Map<Integer, String> classifiers = readEnumCSV("classifiers2.csv");
	public static final int RISK_SITUATION = 1;
	public static final int NORMAL_SITUATION = 2;
	
	public static void main(String[] args) throws IOException, InterruptedException {
		System.out.println("");
		Scanner s = new Scanner(System.in);
		// --------------------------------------------랜덤포레스트 모델학습--------------------------------------------------
		RandomForest RaF = initRF();
//		RaF.createRF();

		// --------------------------------------------딥러닝 모델학습--------------------------------------------------
		MultiLayerNetwork model = initDP();
		
		//---------------------------------------------실시간 검지 -----------------------------------------------------
		new Thread(new Runnable() {
			@Override
			public void run() {
				int count = 0;
				while (true) {
					System.out.println((count + 1) + "st test(attribute 6개 input)");
					int[] attributes_rf = new int[6];
					double[] attributes_dp = new double[6];
					for (int i = 0; i < attributes_dp.length; i++) {
						String temp = s.next();
						if (temp.equals("q")) {
							System.out.println("종료합니다.");
							System.exit(0);
						}
						attributes_rf[i] = (int) Double.parseDouble(temp);
						attributes_dp[i] = Double.parseDouble(temp);
					}
					// 랜포 검지
					int result_rf = RaF.realtime_Start(attributes_rf);

					// 인공신경망 검지
					INDArray testData = Nd4j.create(attributes_dp);
					Map<String, Object> risky = realtime_changeDataForamt(testData);
					normalizer.transform(testData);
					INDArray output = model.output(testData);
					int result_dp = setFittedClassifiers(output, risky);
					logRisky(risky);
					
					if (result_rf == result_dp && result_rf == 1) {
						System.out.println("최종 검지 : 위험");
					} else if (result_rf == result_dp && result_rf == 2) {
						System.out.println("최종 검지 : 일반");
					} else {
						System.out.println("최종 검지 : 의심");
					}
					count++;
					// TODO Auto-generated method stub
				}
			}
		}).start();

	}

	private static MultiLayerNetwork initDP() throws IOException, InterruptedException {
		int labelIndex = 6;
		int numClasses = 2;

		int batchSizeTraining = 320;
		DataSet trainingData = readCSVDataset("dp_Training_certain.csv", batchSizeTraining, labelIndex, numClasses);
		System.out.println(trainingData.getFeatureMatrix());
		normalizer.fit(trainingData);
		normalizer.transform(trainingData);

		final int numInputs = 6;
		int outputNum = 2;
		int iterations = 1000; //
		long seed = 9;

		log.info("Build model....");
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed) //Random number generator seed. Used for reproducability between runs
				.iterations(iterations) // 최적화 반복
				.activation(Activation.LEAKYRELU) // 활성화 함수
				.weightInit(WeightInit.XAVIER) // 초기가중치
				.learningRate(0.1) // 학습률,Learning rate. Defaults to 1e-1
				.list() // Create a ListBuilder (for creating a MultiLayerConfiguration)
				.layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(2).build())
				.layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build())
				.layer(2, new DenseLayer.Builder().nIn(2).nOut(2).build())
				.layer(3,
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
								.activation(Activation.SOFTMAX).nIn(2).nOut(outputNum).build())
				.backprop(true) // 역전파
				.pretrain(false) // 사전학습
				.build();

		// run the model
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init(); // Initialize the MultiLayerNetwork. This should be called once before the network is used.
		model.setListeners(new ScoreIterationListener(100)); // 설정된 iteration 마다  원하는 행동들을 해서  트레이닝 과정을 볼수  있다.

		model.fit(trainingData);
		return model;
	}

	private static RandomForest initRF() {
		String traindata = "rf_Training_certain.csv"; // 훈련데이터 Path
		// String testdata = "TrainingData_driverHabit_park_rf.csv"; //테스트데이터
		int numTrees = 50;// 트리 수

		// 훈련데이터 읽기
		DescribeTrees DT = new DescribeTrees(traindata);
		ArrayList<int[]> Input = DT.CreateInput(traindata);

		int categ = 2;

		// 테스트데이터 읽기
		// DescribeTrees DTT = new DescribeTrees(testdata);
		// ArrayList<int[]> Test = DTT.CreateInput(testdata);
		int[] Test = new int[] { 56, 0, -28, -46, 28, 3 };

		// RandomForest RaF = new RandomForest(numTrees, Input, Test);
		RandomForest RaF = new RandomForest(numTrees, Input);

		// C : 범주 수 , M : 범주 속성
		RaF.C = categ;
		RaF.M = Input.get(0).length - 1;
		RaF.Ms = (int) Math.round(Math.log(RaF.M) / Math.log(2) + 1);
		return RaF;
	}

	public static void logRisky(Map<String, Object> risky) {
		// for (Map<String, Object> r : risky.values())
		// log.info(r.toString());

		System.out.print("인공신경망 검지 : [before:" + risky.get("before") + ", after:" + risky.get("after") + ", avg_decel:"
				+ risky.get("avg_decel") + ", max_decel:" + risky.get("max_decel") + ", distance:"
				+ risky.get("distance") + ", second:" + risky.get("second") + "]");
		if (risky.get("classifier").equals("abnormal-risky")) {
			System.out.println("-위험");
		} else {
			System.out.println("-일반");
		}

	}
	
	public static int setFittedClassifiers(INDArray output, Map<String, Object> risky) {

		String temp = classifiers.get(maxIndex(getFloatArrayFromSlice(output)));
		risky.put("classifier", temp);
		if (temp.equals("abnormal-risky")) {
			return RISK_SITUATION;
		} else {
			return NORMAL_SITUATION;
		}
	}

	public static float[] getFloatArrayFromSlice(INDArray rowSlice) {
		float[] result = new float[rowSlice.columns()];//rowSlice.columns() 범주 수
		for (int i = 0; i < rowSlice.columns(); i++) {
			result[i] = rowSlice.getFloat(i);
		}
		
		return result;
	}

	public static int maxIndex(float[] vals) { //vals는 범주수
		int maxIndex = 0;
		for (int i = 1; i < vals.length; i++) {
			float newnumber = vals[i];
			if ((newnumber > vals[maxIndex])) {
				maxIndex = i;
			}
		}
		return maxIndex;
	}

	public static Map<Integer, Map<String, Object>> changeDataForamt(DataSet testData) {
		Map<Integer, Map<String, Object>> risky = new HashMap<>();
		INDArray features = testData.getFeatureMatrix();
		for (int i = 0; i < features.rows(); i++) {
			INDArray slice = features.slice(i);
			Map<String, Object> driving_information = new HashMap<>();

			// set the attributes
			driving_information.put("before", slice.getDouble(0));
			driving_information.put("after", slice.getDouble(1));
			driving_information.put("avg_decel", slice.getDouble(2));
			driving_information.put("max_decel", slice.getDouble(3));
			driving_information.put("distance", slice.getDouble(4));
			driving_information.put("second", slice.getDouble(5));
			risky.put(i, driving_information);
		}
		return risky;
	}

	public static Map<String, Object> realtime_changeDataForamt(INDArray testData) {
		Map<String, Object> driving_information = new HashMap<>();
		driving_information.put("before", testData.getDouble(0));
		driving_information.put("after", testData.getDouble(1));
		driving_information.put("avg_decel", testData.getDouble(2));
		driving_information.put("max_decel", testData.getDouble(3));
		driving_information.put("distance", testData.getDouble(4));
		driving_information.put("second", testData.getDouble(5));
		return driving_information;
	}
	
	public static Map<Integer, String> readEnumCSV(String csvFileClasspath) {
		try {
			List<String> lines = IOUtils.readLines(new FileInputStream(csvFileClasspath));
			Map<Integer, String> enums = new HashMap<>();
			for (String line : lines) {
				String[] parts = line.split(",");
				enums.put(Integer.parseInt(parts[0]), parts[1]);
			}
			return enums;
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}

	}

	private static DataSet readCSVDataset(String csvFileClasspath, int batchSize, int labelIndex, int numClasses)
			throws IOException, InterruptedException {

		RecordReader rr = new CSVRecordReader();
		rr.initialize(new FileSplit(new File(csvFileClasspath)));
		DataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses);
		return iterator.next();
	}
}
