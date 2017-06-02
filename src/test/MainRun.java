package test;

import java.util.ArrayList;
import java.util.Arrays;

public class MainRun {
	@SuppressWarnings("static-access")
	public static void main(String args[]) {
		
		String traindata = "rf_Training_certain.csv"; //훈련데이터 Path
		String testdata = "rf_Test_certain.csv"; //테스트데이터 Path
		int numTrees = 100;
		
		
		//훈련데이터 읽기
		DescribeTrees DT = new DescribeTrees(traindata);
		ArrayList<int[]> Input = DT.CreateInput(traindata);  
		
		System.out.println(Input);
		int categ = 2;
		
		//테스트데이터 읽기
		DescribeTrees DTT = new DescribeTrees(testdata);
		ArrayList<int[]> Test = DTT.CreateInput(testdata);
				
		RandomForest RaF = new RandomForest(numTrees, Input, Test);
		
		//C : 범주 수 , M : 범주 속성 
		RaF.C = categ;
		RaF.M = Input.get(0).length - 1;
		RaF.Ms = (int) Math.round(Math.log(RaF.M) / Math.log(2) + 1);
		

		RaF.Start();
		
	}
}
