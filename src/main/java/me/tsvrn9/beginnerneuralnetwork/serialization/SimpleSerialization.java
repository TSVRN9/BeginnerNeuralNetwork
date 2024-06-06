package me.tsvrn9.beginnerneuralnetwork.serialization;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Modifier;

public class SimpleSerialization {
    private static final Gson gson = new GsonBuilder().serializeNulls().excludeFieldsWithModifiers(Modifier.STATIC).setPrettyPrinting().create();
    public static void save(Object obj, String filePath) {
        try (FileWriter writer = new FileWriter(filePath)) {
            gson.toJson(obj, writer);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static <T> T load(String filePath, Class<T> clazz) {
        T data = null;

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            data = gson.fromJson(reader, clazz);
        } catch (IOException e) {
            e.printStackTrace();
        }

        return data;
    }
}
