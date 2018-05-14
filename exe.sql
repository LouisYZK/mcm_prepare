drop procedure if exists chaxun;
delimiter $
create procedure chaxun(out s int)
	begin
		select count(*) into s from data where sex =1;
		select s;
	end 
	$
delimiter ;

-- 编写存储过程，根据肺活量划分等级；
alter table data add column class_fhl varchar(10);
drop procedure if exists clf;
delimiter $
create procedure clf()
	begin
		-- 声明变量
		declare fhl_exe int(10);
		declare sex_exe int(2);
		declare xuehao_exe varchar(20);
		-- 创建游标结束标志
		declare done int default false;
		-- 创建游标
		declare cur_pass cursor for select xuehao,fhl from data;
		-- 绑定游标结束标志
		declare continue handler for not found set done = true;
		-- 打开游标
		open cur_pass;
		-- 循环游标
		read_loop:loop
			-- 从游标取值
			fetch next from cur_pass into xuehao_exe,fhl_exe;
			if  fhl_exe >4000 then
				update data set class_fhl = "HIGH" where xuehao = xuehao_exe;
			else
				update data set class_fhl = "LOW" where xuehao= xuehao_exe;
			end if ;
			select class_fhl  from data where xuehao = xuehao_exe;
			if done then
				leave read_loop;
			end if;
		end loop;
			select class_fhl from data;
	end $
delimiter ;