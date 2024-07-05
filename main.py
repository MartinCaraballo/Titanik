import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Se crea un dataframe utilizando el archivo CSV provisto.
df = pd.read_csv('titanik.csv')


# Completa la columna 'age' con los datos correspondientes.
def populate_age_column():
    mean_age_male = df[df['sex'] == 'male']['age'].mean()
    mean_age_female = df[df['sex'] == 'female']['age'].mean()

    df.loc[
        (df['age'].isnull()) & (df['sex'] == 'male'),
        'age'
    ] = mean_age_male

    df.loc[
        (df['age'].isnull()) & (df['sex'] == 'female'),
        'age'
    ] = mean_age_female


# Calcula las estadisticas requeridas utilizando los valores de la columna
# 'age'.
def calculate_age_statistics():
    mean_age = df['age'].mean()
    median_age = df['age'].median()
    mode_age = df['age'].mode()[0]
    range_age = df['age'].max() - df['age'].min()
    var_age = df['age'].var()
    std_age = df['age'].std()


# Calcula la tasa de supervivencia general.
def calculate_survival_rate():
    s_rate = df['survived'].mean()
    print(f"Tasa de supervivencia general: {s_rate}")


# Calcula la tasa de supervivencia para ambos generos.
def calculate_survival_rate_by_gender():
    survival_rate_male = df[df['sex'] == 'male']['survived'].mean()
    survival_rate_female = df[df['sex'] == 'female']['survived'].mean()

    print(f"Tasa de supervivencia de hombres: {survival_rate_male}")
    print(f"Tasa de supervivencia de mujeres: {survival_rate_female}")


# Genera un histograma con las edades de los pasajeros por clase.
def make_age_histogram_by_class():
    df[df['pclass'] == 1]['age'].hist(alpha=0.5, label='Primera Clase')
    df[df['pclass'] == 2]['age'].hist(alpha=0.5, label='Segunda Clase')
    df[df['pclass'] == 3]['age'].hist(alpha=0.5, label='Tercera Clase')
    plt.legend()
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')
    plt.title('Histograma de Edades por Clase')
    plt.show()


# Genera el diagrama de cajas con las edades de los supervivientes y otro para
# los no supervivientes.
def make_box_diagram():
    df.boxplot(column='age', by='survived')
    plt.xlabel('Supervivencia')
    plt.ylabel('Edad')
    plt.title('Diagrama de Cajas de Edades por Supervivencia')
    plt.show()


# Crea un intervalo de confianza del 95% para la edad promedio de las personas
# en el barco.
def create_trust_interval():
    mean_age = df['age'].mean()
    std_err_age = stats.sem(df['age'])

    conf_inter = stats.t.interval(
        0.95, len(df['age']) - 1,
        loc=mean_age,
        scale=std_err_age
    )

    print(
        f"Intervalo de confianza del 95% para la edad promedio: {conf_inter}")


# Realiza una prueba T para saber si la edad promedio de las mujeres
# interesadas en abordar al Titanic es nayor a 56.
def calculate_avg_women_age_t_test():
    ttest_result_women = stats.ttest_1samp(
        df[df['sex'] == 'female']['age'],
        56
    )

    print(f"Resultado de la prueba T para mujeres: {ttest_result_women}")


# Realiza una prueba T para saber si la edad promedio de los hombres
# interesados en abordar al Titanic es nayor a 56.
def calculate_avg_men_age_t_test():
    ttest_result_men = stats.ttest_1samp(
        df[df['sex'] == 'male']['age'],
        56
    )

    print(f"Resultado de la prueba T para hombres: {ttest_result_men}")


# Calcula la diferencia entre las tasas de supervivencia de ambos generos.
def calculate_gender_survival_rate_difference():
    survival_rate_male = df[df['sex'] == 'male']['survived']
    survival_rate_female = df[df['sex'] == 'female']['survived']
    ttest_gender_survival = stats.ttest_ind(survival_rate_male,
                                            survival_rate_female,
                                            equal_var=False)
    print(
        f"Resultado de la prueba T para tasas de supervivencia por género: "
        f"{ttest_gender_survival}"
    )


# Calcula la diferencia entre las tasas de supervivencia de todas las clases.
def calculate_class_survival_rate_difference():
    survival_rate_class1 = df[df['pclass'] == 1]['survived']
    survival_rate_class2 = df[df['pclass'] == 2]['survived']
    survival_rate_class3 = df[df['pclass'] == 3]['survived']
    ttest_class_survival = stats.f_oneway(survival_rate_class1,
                                          survival_rate_class2,
                                          survival_rate_class3)
    print(
        f"Resultado de la prueba ANOVA para tasas de supervivencia por clase: "
        f"{ttest_class_survival}"
    )


# Realiza una prueba T para comprobar si en promedio las mujeres tenian menos
# edad que los hombres.
def calculate_t_test_age_difference_by_gender():
    stats.ttest_ind(df[df['sex'] == 'female']['age'],
                    df[df['sex'] == 'male']['age'], equal_var=False)


def main():
    populate_age_column()


if __name__ == "__main__":
    main()
