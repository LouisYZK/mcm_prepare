% 三次拟合实例
year = [1625,1830,1930,1960,1974,1987,1999]
population = [5,10,20,30,40,50,60]
year_1 = 1625:2020
year_2 = 2000:2020
[coef,bias] = polyfit(year,population,3) 
population_1 = polyval(coef,year_1)
population_2 = polyval(coef,year_2)
plot(year,population,'*',year_2,population_2,'X',year_1,population_1)
legend('实际数据','拟合数据')
xlabel('年份')
ylabel('人口')