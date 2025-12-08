```markdown
# JAVAAA

This repository is a minimal plain-Java (javac) scaffold created to demonstrate how to build
and run a project without Maven or Gradle.

Build and run:

```bash
# compile
bash build.sh

# run
bash run.sh

# run with arguments
bash run.sh hello world
```

Files added by the scaffold:
- `src/com/example/App.java` — sample `main` program
- `build.sh` — compiles `.java` files into the `out/` directory
- `run.sh` — runs the sample `App` class
- `.gitignore` — ignores `out/`
- `pom.xml` — optional Maven POM that uses the existing `src/` layout

See the scripts for details.

Optional: build with Maven

```bash
# compile with maven
mvn -B package

# run the produced classes (or use `bash run.sh`)
java -cp target/classes com.example.App
```
```
# JAVAAA