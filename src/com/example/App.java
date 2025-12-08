package com.example;

public class App {
    public static void main(String[] args) {
        System.out.println("Hello from plain Java scaffold!");
        if (args.length > 0) {
            System.out.println("Args:");
            for (String a : args) {
                System.out.println(" - " + a);
            }
        }
    }
}
