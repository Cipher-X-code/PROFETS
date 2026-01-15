// write a java program to accept  infinite num from a user and calculate the count of positive  negative  your sentinel num

import java.util.Scanner;
public class SentinelControlLoop{
	public static void main (String [] args ){
		Scanner input = new Scanner (System.in);
		int num ;
		int positiveCount  =  0;
		int NegativeCount = 0;
		
		System.out.print("Enter a  number: "); // display a message 
		num = input.nextInt;
		
		if (num == -1);
		System.out.print(" terminate when you enter - 1");
		
		{ while (true) {
			System.out.print(" Enter your new  number :");
			num = input.nextInt; // receive input from the user .
			
			if ( num == -1 ){
			break;
		} else if (num > 0){
			positiveCount++;
			break;
		} else if (num < 0){
			negativeCount++;
			break;
		}
		}
	
			
			System.out.printf ("Count of positiveCount %d %n ", positiveCount);
			System.out.printf ("Count of positiveCount %d %n ", negativeCountCount);
			System.out.println("Sentinel number used: -1");	
		}
	}
}
