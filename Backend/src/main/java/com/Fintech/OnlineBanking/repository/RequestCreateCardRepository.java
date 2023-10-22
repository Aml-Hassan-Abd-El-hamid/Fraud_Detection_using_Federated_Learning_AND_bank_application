package com.Fintech.OnlineBanking.repository;

import java.util.List;



import org.springframework.data.jpa.repository.JpaRepository;

import com.Fintech.OnlineBanking.domain.RequestCreateCard;
import com.Fintech.OnlineBanking.domain.Transactions;

public interface RequestCreateCardRepository extends JpaRepository <RequestCreateCard, Integer>{

	RequestCreateCard findFirstByOrderByRequestNumberCreateCardDesc();

	
}
